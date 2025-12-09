
import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from runtime.feedback_package import TaskType, _get_task_identity
from error_explanation import ErrorExplanationResult


load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment or .env")

OPENAI_MODEL = "gpt-4o"
openai = OpenAI(api_key=OPENAI_API_KEY)



@dataclass
class PatchGenerationResult:
    """
    Output of Step 5.1 Patch Generation.

    This corresponds to:
      - Take current code_t + error_explanation_t + task spec
      - Produce a patched version of the code (code_{t+1})
      - Describe whether we edited faulty lines or rewrote the function.
    """
    benchmark: str
    task_id: str
    model: str

    original_code: str
    patched_code: str

    patch_strategy: str   
    rationale: str      

    raw_response: Dict[str, Any]



# Generate_patched_code_with_openai
def generate_patched_code_with_openai(
    task: TaskType,
    current_code: str,
    error_expl: ErrorExplanationResult,
    model: str = OPENAI_MODEL,
) -> PatchGenerationResult:
    benchmark, tid = _get_task_identity(task)
    problem_spec = task.build_prompt()
    explanation = error_expl.explanation
    diag = getattr(error_expl, "diagnosis", None)

    diagnostic_summary = "No structured diagnosis available."
    if diag is not None:
        parts = []
        if getattr(diag, "failing_behaviour", "").strip():
            parts.append(f"- Failing behaviour: {diag.failing_behaviour.strip()}")
        if getattr(diag, "suspected_bug_location", "").strip():
            parts.append(f"- Suspected bug location: {diag.suspected_bug_location.strip()}")
        if getattr(diag, "what_is_wrong", "").strip():
            parts.append(f"- What is wrong: {diag.what_is_wrong.strip()}")
        if getattr(diag, "why_it_is_wrong", "").strip():
            parts.append(f"- Why it is wrong: {diag.why_it_is_wrong.strip()}")
        if parts:
            diagnostic_summary = "\n".join(parts)

    system_msg = (
        "You are a highly reliable coding assistant. "
        "Your job in THIS STEP is to FIX the existing Python solution code. "
        "You must use the problem specification and the previous error analysis "
        "to correct the logic while preserving the required function signatures "
        "and overall task interface. Focus on minimal, precise edits when possible."
    )

    user_msg = f"""
    # (1) PROBLEM SPECIFICATION
    {problem_spec}

    # (2) CURRENT PYTHON CODE (code_t)
    {current_code}

    (3) ERROR EXPLANATION (error_explanation_t)
    {explanation}
    (4) STRUCTURED DIAGNOSIS (optional)
    {diagnostic_summary}

    YOUR TASK (Step 5.1  Patch Generation)
    You must now produce a fixed version of the Python solution.
    Guidelines:
    If possible, only edit the faulty lines and keep the rest of the code unchanged.
    If the logic is badly broken, you may rewrite the entire function or module.
    You MUST preserve all required function signatures and the overall interface expected by the tests.
    The patched code MUST be valid Python code and should be ready to execute.
    Do NOT include any test code or main-guard unless the problem explicitly requires a full program (e.g., some APPS tasks).
    Do NOT add unnecessary features or refactors; focus on correctness and clarity.
    RESPONSE FORMAT (MANDATORY)
    Respond ONLY as a JSON object with the following fields:

    {{
    "patched_code": "The full, corrected Python code. Do NOT wrap in backticks or Markdown fences.",
    "patch_strategy": "Short description of whether you edited a few lines or rewrote the function/module.",
    "rationale": "Brief explanation of how the changes fix the bug, referring to the error explanation when useful."
    }}

    Rules:
    The 'patched_code' field must contain only Python code, no backticks, no Markdown.

    Do not change the required function names or signatures.

    Do not mention these instructions or the JSON format in the output.
    """

    response = openai.chat.completions.create(
    model=model,
    response_format={"type": "json_object"},
    messages=[
    {"role": "system", "content": system_msg},
    {"role": "user", "content": user_msg},
    ],
    temperature=0.2,
    )

    content = response.choices[0].message.content

    try:
      parsed = json.loads(content)
    except json.JSONDecodeError:
      parsed = {
      "patched_code": content,
      "patch_strategy": "unknown (model returned non-JSON; treated whole response as patched_code)",
      "rationale": "",
      }

    patched_code = parsed.get("patched_code", "").strip()
    patch_strategy = parsed.get("patch_strategy", "").strip()
    rationale = parsed.get("rationale", "").strip()

    try:
        raw_resp: Dict[str, Any] = response.model_dump()
    except Exception:
      raw_resp = {"raw": str(response)}

    return PatchGenerationResult(
    benchmark=benchmark,
    task_id=tid,
    model=model,
    original_code=current_code,
    patched_code=patched_code,
    patch_strategy=patch_strategy,
    rationale=rationale,
    raw_response=raw_resp,
  )


def produce_next_code_version(
task: TaskType,
current_code: str,
error_expl: ErrorExplanationResult,
) -> Tuple[str, PatchGenerationResult]:

  patch_info = generate_patched_code_with_openai(
      task=task,
      current_code=current_code,
      error_expl=error_expl,
  )

  next_code = patch_info.patched_code
  return next_code, patch_info
