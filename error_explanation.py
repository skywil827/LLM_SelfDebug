from dataclasses import dataclass
from typing import Any, Dict
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from runtime.feedback_package import FeedbackPackage, TaskType, _get_task_identity

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

openai = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL = "gpt-4o-mini" 


@dataclass
class ErrorExplanationIO:
    """
    Container for everything the LLM should see in Step 4.1:
      - Original problem spec (problem + tests)
      - Current code_t
      - Feedback_t (formatted text block)
    """
    benchmark: str
    task_id: str

    problem_spec: str      
    current_code: str     
    feedback_block: str    


@dataclass
class BugDiagnosisResult:
    """
    Output of Step 4.2 – Diagnose the bug.

    This does NOT change the code, it just explains:
      - observed failing behaviour
      - where the bug likely lives
      - what is wrong
      - why it is wrong w.r.t. the problem requirements
    """
    benchmark: str
    task_id: str
    model: str

    failing_behaviour: str          
    suspected_bug_location: str      
    what_is_wrong: str               
    why_it_is_wrong: str            

    raw_response: Dict[str, Any]

from dataclasses import dataclass

@dataclass
class ErrorExplanationResult:
    """
    Output of Step 4: error_explanation_t

    A single natural-language explanation that summarizes:
      - what is going wrong
      - where it is happening
      - why it violates the problem requirements
    """
    benchmark: str
    task_id: str
    model: str
    explanation: str
    diagnosis: BugDiagnosisResult



def build_error_explanation_io(
    task: TaskType,
    current_code: str,
    feedback: FeedbackPackage,
) -> ErrorExplanationIO:
    benchmark, tid = _get_task_identity(task)

    problem_spec = task.build_prompt()
    feedback_block = feedback.to_model_feedback_block()

    return ErrorExplanationIO(
        benchmark=benchmark,
        task_id=tid,
        problem_spec=problem_spec,
        current_code=current_code,
        feedback_block=feedback_block,
    )


def diagnose_bug_with_openai(
    io: ErrorExplanationIO,
    model: str = OPENAI_MODEL,
) -> BugDiagnosisResult:

    benchmark = io.benchmark
    tid = io.task_id

    system_msg = (
        "You are a highly skilled debugging assistant. "
        "You receive: (1) the original programming problem, "
        "(2) the current Python solution code, and "
        "(3) execution feedback (test results, errors, stdout/stderr). "
        "Your job in THIS STEP is ONLY to diagnose the bug: "
        "describe what behaviour is wrong, where the bug likely is, "
        "what is logically incorrect, and why this violates the problem requirements. "
        "Do NOT propose or write any new code in this step."
    )

    user_msg = f"""
        # (1) PROBLEM SPECIFICATION
        {io.problem_spec}

        # (2) CURRENT PYTHON CODE
        {io.current_code}

        # (3) EXECUTION FEEDBACK
        {io.feedback_block}

        # YOUR TASK (Step 4.2 – Bug Diagnosis Only)

        You MUST:

        1. Map the failing tests and/or error messages to specific parts of the code.
        2. Identify what is wrong in the logic, such as:
        - Wrong condition
        - Wrong formula or arithmetic
        - Missing base case
        - Wrong loop bounds
        - Incorrect handling of edge cases
        - Misuse of data structures
        3. Explain WHY this is wrong, explicitly referencing the problem requirements.

        RESPONSE FORMAT (MANDATORY)
        Respond ONLY as a JSON object with the following fields:

        {{
        "failing_behaviour": "Summarize what the code currently does wrong based on the feedback (tests / errors).",
        "suspected_bug_location": "Describe which function, block, or lines are most likely responsible for the bug.",
        "what_is_wrong": "Describe the logical or algorithmic mistake (wrong condition, formula, missing case, etc.).",
        "why_it_is_wrong": "Explain why this behaviour violates the problem's requirements, using those requirements explicitly."
        }}

        Rules:
        - Do NOT include any Python code in the JSON fields.
        - Do NOT include backticks or Markdown fences inside the JSON.
        - If all tests passed and you see no clear bug, clearly say that in all fields.
    """

    response = openai.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",  "content": user_msg},
        ],
        temperature=0.1,
    )

    content = response.choices[0].message.content

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
    
        parsed = {
            "failing_behaviour": content,
            "suspected_bug_location": "",
            "what_is_wrong": "",
            "why_it_is_wrong": "",
        }

    failing_behaviour = parsed.get("failing_behaviour", "").strip()
    suspected_bug_location = parsed.get("suspected_bug_location", "").strip()
    what_is_wrong = parsed.get("what_is_wrong", "").strip()
    why_it_is_wrong = parsed.get("why_it_is_wrong", "").strip()

    try:
        raw_resp: Dict[str, Any] = response.model_dump()
    except Exception:
        raw_resp = {"raw": str(response)}

    return BugDiagnosisResult(
        benchmark=benchmark,
        task_id=tid,
        model=model,
        failing_behaviour=failing_behaviour,
        suspected_bug_location=suspected_bug_location,
        what_is_wrong=what_is_wrong,
        why_it_is_wrong=why_it_is_wrong,
        raw_response=raw_resp,
    )


def build_error_explanation_text(
    io: ErrorExplanationIO,
    diag: BugDiagnosisResult,
) -> ErrorExplanationResult:
    if (
        "all tests passed" in diag.failing_behaviour.lower()
        or "no clear bug" in diag.failing_behaviour.lower()
        or (diag.failing_behaviour.strip() == "" and diag.what_is_wrong.strip() == "")
    ):
        explanation = (
            "According to the current execution feedback, all visible tests passed and "
            "no clear failure was observed. The code appears to satisfy the provided "
            "test cases, although hidden edge cases may still exist beyond the given feedback."
        )
    else:
        parts = []

        if diag.failing_behaviour.strip():
            parts.append(
                "First, based on the execution feedback, the observed failing behaviour is:\n"
                f"- {diag.failing_behaviour.strip()}"
            )

        if diag.suspected_bug_location.strip():
            parts.append(
                "\nThe bug is most likely located in the following part of the code:\n"
                f"- {diag.suspected_bug_location.strip()}"
            )

        if diag.what_is_wrong.strip():
            parts.append(
                "\nIn terms of logic, the core issue can be summarized as:\n"
                f"- {diag.what_is_wrong.strip()}"
            )

        if diag.why_it_is_wrong.strip():
            parts.append(
                "\nThis is incorrect with respect to the problem requirements because:\n"
                f"- {diag.why_it_is_wrong.strip()}"
            )
        if not parts:
            parts.append(
                "The diagnosis did not yield a specific bug description. "
                "There may not be enough information in the feedback to identify the failure."
            )

        explanation = "\n".join(parts)

    return ErrorExplanationResult(
        benchmark=diag.benchmark,
        task_id=diag.task_id,
        model=diag.model,
        explanation=explanation,
        diagnosis=diag,
    )

