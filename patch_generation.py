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
    benchmark: str
    task_id: str
    model: str

    original_code: str
    patched_code: str

    patch_strategy: str
    rationale: str

    raw_response: Dict[str, Any]


def _is_swe_benchmark(benchmark: str) -> bool:
    b = (benchmark or "").strip()
    return b.upper().startswith("SWE")


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
        fb = getattr(diag, "failing_behaviour", "")
        loc = getattr(diag, "suspected_bug_location", "")
        wrong = getattr(diag, "what_is_wrong", "")
        why = getattr(diag, "why_it_is_wrong", "")
        if str(fb).strip():
            parts.append(f"- Failing behaviour: {str(fb).strip()}")
        if str(loc).strip():
            parts.append(f"- Suspected bug location: {str(loc).strip()}")
        if str(wrong).strip():
            parts.append(f"- What is wrong: {str(wrong).strip()}")
        if str(why).strip():
            parts.append(f"- Why it is wrong: {str(why).strip()}")
        if parts:
            diagnostic_summary = "\n".join(parts)

    is_swe = _is_swe_benchmark(benchmark)

    if is_swe:
        system_msg = (
            "You are a highly reliable software engineer working on a repository. "
            "Your job in THIS STEP is to FIX the existing candidate unified diff PATCH. "
            "Use the bug report prompt and the error analysis to produce a corrected patch. "
            "Keep it minimal and ensure it is a valid unified diff suitable for `git apply`."
        )

        response_schema = """
        {
          "patched_patch": "The full, corrected unified diff patch. Do NOT wrap in backticks or Markdown fences.",
          "patch_strategy": "Short description of what you changed in the patch.",
          "rationale": "Brief explanation of how the updated patch addresses the failure."
        }
        """
        artifact_field = "patched_patch"
        artifact_label = "CURRENT PATCH (patch_t)"
        rules_line = (
            "The 'patched_patch' field must contain ONLY a unified diff patch starting with "
            "'diff --git a/... b/...'. No backticks. No Markdown. No prose outside the patch."
        )
        additional_guidelines = """
        Guidelines:
        - Preserve unified diff structure: diff --git, --- a/..., +++ b/..., @@ hunks.
        - Do NOT include any explanation text inside the patch itself.
        - If the previous failure is 'InvalidPatch', correct the diff formatting first.
        - Keep changes small and focused.
        """
    else:
        system_msg = (
            "You are a highly reliable coding assistant. "
            "Your job in THIS STEP is to FIX the existing Python solution code. "
            "Use the problem specification and the error analysis to correct the logic "
            "while preserving the required function signatures and task interface. "
            "Focus on minimal, precise edits when possible."
        )

        response_schema = """
        {
          "patched_code": "The full, corrected Python code. Do NOT wrap in backticks or Markdown fences.",
          "patch_strategy": "Short description of whether you edited a few lines or rewrote the function/module.",
          "rationale": "Brief explanation of how the changes fix the bug, referring to the error explanation when useful."
        }
        """
        artifact_field = "patched_code"
        artifact_label = "CURRENT PYTHON CODE (code_t)"
        rules_line = "The 'patched_code' field must contain ONLY Python code. No backticks. No Markdown."
        additional_guidelines = """
        Guidelines:
        - Preserve required function names/signatures.
        - Fix only what is necessary for correctness.
        - Avoid refactors unrelated to the failure.
        - Do NOT include test code unless explicitly required by the problem.
        """

    user_msg = f"""
        # (1) PROBLEM SPECIFICATION
        {problem_spec}

        # (2) {artifact_label}
        {current_code}

        # (3) ERROR EXPLANATION (error_explanation_t)
        {explanation}

        # (4) STRUCTURED DIAGNOSIS (optional)
        {diagnostic_summary}

        {additional_guidelines}

        RESPONSE FORMAT (MANDATORY)
        Respond ONLY as a JSON object with the following fields:

        {response_schema}

        Rules:
        - {rules_line}
        - Do not mention these instructions or the JSON format in the output.
        """.strip()

    response = openai.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    content = response.choices[0].message.content

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {
            artifact_field: content,
            "patch_strategy": "unknown (model returned non-JSON; treated whole response as artifact)",
            "rationale": "",
        }

    patched_artifact = str(parsed.get(artifact_field, "") or "").strip()
    patch_strategy = str(parsed.get("patch_strategy", "") or "").strip()
    rationale = str(parsed.get("rationale", "") or "").strip()

    try:
        raw_resp: Dict[str, Any] = response.model_dump()
    except Exception:
        raw_resp = {"raw": str(response)}

    return PatchGenerationResult(
        benchmark=benchmark,
        task_id=tid,
        model=model,
        original_code=current_code,
        patched_code=patched_artifact,
        patch_strategy=patch_strategy,
        rationale=rationale,
        raw_response=raw_resp,
    )


def produce_next_code_version(
    task: TaskType,
    current_code: str,
    error_expl: ErrorExplanationResult,
    model: str = OPENAI_MODEL,
) -> Tuple[str, PatchGenerationResult]:
    """
    Returns: (next_code_or_patch, patch_info)
    model: which OpenAI model should perform the patch step
    """
    patch_info = generate_patched_code_with_openai(
        task=task,
        current_code=current_code,
        error_expl=error_expl,
        model=model,
    )
    next_code = patch_info.patched_code
    return next_code, patch_info
