import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai
from google.generativeai import GenerativeModel
import anthropic
from humanEvalInput import HumanEvalTask, load_humaneval_task
from MBPPInput import MBPPTask, load_mbpp_task
from APPSInput import APPSTask, load_apps_task
from runtime.feedback_package import build_feedback_package
from runtime.code_exec import ExecutionResult, execute_task_code
from error_explanation import (
    build_error_explanation_io,
    diagnose_bug_with_openai,
    build_error_explanation_text,
)
from patch_generation import produce_next_code_version
from datasets import load_dataset
from human_eval.data import read_problems
import matplotlib.pyplot as plt
import difflib 


# Environment & Clients
load_dotenv(override=True)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

openai = OpenAI()
google.generativeai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Safe defaults for models
OPENAI_MODEL = "gpt-4o-mini"
GOOGLE_MODEL = "gemini-2.0-flash"


@dataclass
class InitialCodeResult:
    benchmark: str
    task_id: str
    model: str
    plan: str
    code: str
    explanation: str
    raw_response: dict
    raw_prompt: str


TaskType = Union[HumanEvalTask, MBPPTask, APPSTask]


def _get_task_identity(task: TaskType) -> tuple[str, str]:
    """
    Helper: extract (benchmark, id_string) from any of the task types.
    """
    benchmark = task.constraints.get("benchmark", "UNKNOWN")

    if isinstance(task, HumanEvalTask):
        tid = task.task_id
    elif isinstance(task, MBPPTask):
        tid = f"MBPP/{task.task_id}"
    elif isinstance(task, APPSTask):
        tid = f"APPS/{task.problem_id}"
    else:
        tid = "UNKNOWN"

    return benchmark, tid


# Initial Code Generation – OpenAI
def generate_initial_code_with_openai(
    task: TaskType,
    model: str = OPENAI_MODEL,
) -> InitialCodeResult:
    benchmark, tid = _get_task_identity(task)
    base_prompt = task.build_prompt()

    system_msg = (
        "You are a highly reliable coding assistant. "
        "Your job is to understand the problem, propose a clear plan, "
        "then write correct and clean Python code, and finally explain your solution."
    )

    user_instructions = f"""
        You will receive a programming task from the {benchmark} benchmark.

        TASK SPECIFICATION
        ------------------
        {base_prompt}

        RESPONSE FORMAT (MANDATORY)
        ---------------------------
        Respond ONLY as a JSON object with the following fields:

        {{
          "plan": "Short step-by-step reasoning about how you will solve the problem.",
          "code": "Python code implementing your solution. Do NOT wrap in ``` fences.",
          "explanation": "A clear explanation of how the code works."
        }}

        Constraints:
        - The "code" MUST be valid Python.
        - Do NOT include backticks or Markdown fences in any field.
        - For HumanEval/MBPP, implement ONLY the required function(s), not a CLI.
        - For APPS, you may write a full program if needed, but keep it minimal and correct.
    """

    response = openai.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_instructions},
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {
            "plan": "",
            "code": "",
            "explanation": content,
        }

    plan = parsed.get("plan", "").strip()
    code = parsed.get("code", "").strip()
    explanation = parsed.get("explanation", "").strip()

    return InitialCodeResult(
        benchmark=benchmark,
        task_id=tid,
        model=model,
        plan=plan,
        code=code,
        explanation=explanation,
        raw_response=response.model_dump() if hasattr(response, "model_dump") else response,
        raw_prompt=user_instructions,
    )


# Initial Code Generation – Gemini
def generate_initial_code_with_gemini(
    task: TaskType,
    model: str = GOOGLE_MODEL,
) -> InitialCodeResult:
    benchmark, tid = _get_task_identity(task)
    base_prompt = task.build_prompt()

    system_msg = (
        "You are a highly reliable coding assistant. "
        "Your job is to understand the problem, propose a clear plan, "
        "then write correct and clean Python code, and finally explain your solution."
    )

    user_instructions = f"""
        You will receive a programming task from the {benchmark} benchmark.

        TASK SPECIFICATION
        ------------------
        {base_prompt}

        RESPONSE FORMAT (MANDATORY)
        ---------------------------
        Respond ONLY as a JSON object with the following fields:

        {{
          "plan": "Short step-by-step reasoning about how you will solve the problem.",
          "code": "Python code implementing your solution. Do NOT wrap in ``` fences.",
          "explanation": "A clear explanation of how the code works."
        }}

        Constraints:
        - The "code" MUST be valid Python.
        - Do NOT include backticks or Markdown fences in any field.
        - For HumanEval/MBPP, implement ONLY the required function(s), not a CLI.
        - For APPS, you may write a full program if needed, but keep it minimal and correct.

        System instructions:
        {system_msg}
    """

    model_obj = GenerativeModel(model)
    response = model_obj.generate_content(user_instructions)

    content = getattr(response, "text", "") or str(response)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {
            "plan": "",
            "code": "",
            "explanation": content,
        }

    plan = parsed.get("plan", "").strip()
    code = parsed.get("code", "").strip()
    explanation = parsed.get("explanation", "").strip()

    raw_response = response.to_dict() if hasattr(response, "to_dict") else {"raw": str(response)}

    return InitialCodeResult(
        benchmark=benchmark,
        task_id=tid,
        model=model,
        plan=plan,
        code=code,
        explanation=explanation,
        raw_response=raw_response,
        raw_prompt=user_instructions,
    )


# Baseline (NO self-debugging)
def run_single_task_no_self_debug(
    task: TaskType,
    provider: str,
    model_name: str,
) -> Dict[str, Any]:
    if provider == "openai":
        init = generate_initial_code_with_openai(task, model=model_name)
    elif provider == "gemini":
        init = generate_initial_code_with_gemini(task, model=model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    candidate_code = init.code
    exec_result = execute_task_code(task, candidate_code)

    return {
        "benchmark": exec_result.benchmark,
        "task_id": exec_result.task_id,
        "provider": provider,
        "model": model_name,
        "mode": "baseline",
        "passed": exec_result.passed,
        "num_tests": exec_result.num_tests,
        "num_passed": exec_result.num_passed,
        "error_type": exec_result.error_type,
        "error_message": exec_result.error_message,
        "num_iterations": 1,
        "self_debug_used": False,
        "initial_model": f"{provider}:{model_name}",
        "patch_model": None,
        "initial_code": candidate_code,
        "final_code": candidate_code,
        "patch_explanations": [],
    }


# Self-Debugging – Single Task
def run_single_task_with_self_debug(
    task: TaskType,
    provider: str,
    model_name: str,
    max_self_debug_iters: int = 3,
) -> Dict[str, Any]:
    if provider == "openai":
        init = generate_initial_code_with_openai(task, model=model_name)
    elif provider == "gemini":
        init = generate_initial_code_with_gemini(task, model=model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    current_code = init.code
    initial_code = current_code
    patch_explanations = []
    num_iterations = 0
    used_self_debug = False
    final_exec_result: Optional[ExecutionResult] = None
    initial_error_type = None
    initial_error_message = None
    initial_num_tests = None
    initial_num_passed = None

    for it in range(max_self_debug_iters + 1):
        num_iterations = it + 1

        exec_result = execute_task_code(task, current_code)
        final_exec_result = exec_result

        if it == 0:
            initial_error_type = exec_result.error_type
            initial_error_message = exec_result.error_message
            initial_num_tests = exec_result.num_tests
            initial_num_passed = exec_result.num_passed

        if exec_result.passed or exec_result.num_tests == 0:
            break

        if it == max_self_debug_iters:
            break

        used_self_debug = True

        feedback = build_feedback_package(task, current_code, exec_result)
        io_bundle = build_error_explanation_io(task, current_code, feedback)
        diag = diagnose_bug_with_openai(io_bundle)
        err_expl = build_error_explanation_text(io_bundle, diag)

        next_code, patch_info = produce_next_code_version(task, current_code, err_expl)
        current_code = next_code

        rationale = getattr(patch_info, "rationale", None)
        if rationale:
            patch_explanations.append(rationale)
        else:
            patch_explanations.append(
                "Patch applied, but no explicit rationale was returned from patch_generation."
            )

    if final_exec_result is None:
        benchmark, tid = _get_task_identity(task)
        final_exec_result = ExecutionResult(
            benchmark=benchmark,
            task_id=tid,
            passed=False,
            num_tests=0,
            num_passed=0,
            error_type="RuntimeError",
            error_message="Self-debug loop did not produce any execution result.",
            traceback_str=None,
            stdout="",
            stderr="",
        )

    return {
        "benchmark": final_exec_result.benchmark,
        "task_id": final_exec_result.task_id,
        "provider": provider,
        "model": model_name,
        "mode": "self_debug",
        "passed": final_exec_result.passed,
        "num_tests": final_exec_result.num_tests,
        "num_passed": final_exec_result.num_passed,
        "error_type": final_exec_result.error_type,
        "error_message": final_exec_result.error_message,
        "num_iterations": num_iterations,
        "self_debug_used": used_self_debug,
        "initial_model": f"{provider}:{model_name}",
        "patch_model": "openai:gpt-4o",
        "initial_code": initial_code,
        "final_code": current_code,
        "patch_explanations": patch_explanations,
        "prompt_to_model": init.raw_prompt,
        "initial_explanation": init.explanation,
        "initial_error_type": initial_error_type,
        "initial_error_message": initial_error_message,
        "initial_num_tests": initial_num_tests,
        "initial_num_passed": initial_num_passed,
    }


# Benchmark runner
def evaluate_benchmark_on_model(
    benchmark: str,
    provider: str,
    model_name: str,
    max_tasks: Optional[int] = None,
    mode: str = "baseline",
    max_self_debug_iters: int = 3,
) -> Dict[str, Any]:
    total_tasks = 0
    total_passed = 0
    per_task = []

    def run_task(task: TaskType) -> Dict[str, Any]:
        if mode == "baseline":
            return run_single_task_no_self_debug(task, provider, model_name)
        elif mode == "self_debug":
            return run_single_task_with_self_debug(
                task, provider, model_name, max_self_debug_iters=max_self_debug_iters
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    if benchmark == "HumanEval":
        problems = read_problems()
        task_ids = list(problems.keys())
        if max_tasks is not None:
            task_ids = task_ids[:max_tasks]

        for tid in task_ids:
            task = load_humaneval_task(tid)
            res = run_task(task)
            per_task.append(res)
            total_tasks += 1
            if res["passed"]:
                total_passed += 1

    elif benchmark == "MBPP":
        ds = load_dataset("mbpp", "sanitized", split="test")
        if max_tasks is not None:
            ds = ds.select(range(max_tasks))

        for row in ds:
            tid = int(row["task_id"])
            task = load_mbpp_task(tid)
            res = run_task(task)
            per_task.append(res)
            total_tasks += 1
            if res["passed"]:
                total_passed += 1

    elif benchmark == "APPS":
        ds = load_dataset("codeparrot/apps", split="test")
        if max_tasks is not None:
            ds = ds.select(range(max_tasks))

        for row in ds:
            pid = int(row["problem_id"])
            task = load_apps_task(pid, split="test")
            res = run_task(task)
            per_task.append(res)
            total_tasks += 1
            if res["passed"]:
                total_passed += 1

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    pass_rate = (total_passed / total_tasks) if total_tasks > 0 else 0.0

    return {
        "benchmark": benchmark,
        "provider": provider,
        "model": model_name,
        "mode": mode,
        "num_tasks": total_tasks,
        "num_passed": total_passed,
        "pass_rate": pass_rate,
        "details": per_task,
    }


if __name__ == "__main__":
    configs = [
        ("openai", "gpt-4o"),
        ("openai", "gpt-4o-mini"),
        ("openai", "gpt-5.1"),
        ("gemini", "gemini-2.0-flash"),
        ("gemini", "gemini-2.5-pro"),
    ]

    # benchmarks = ["HumanEval", "MBPP", "APPS"]
    benchmarks = ["HumanEval", "MBPP"]

    max_tasks = 3
    max_self_debug_iters = 2

    all_results = []

    for provider, model_name in configs:
        for benchmark in benchmarks:
            print(f"\n{benchmark} on {provider}:{model_name} | BASELINE")
            baseline_summary = evaluate_benchmark_on_model(
                benchmark=benchmark,
                provider=provider,
                model_name=model_name,
                max_tasks=max_tasks,
                mode="baseline",
            )

            print(
                f"Baseline: {baseline_summary['num_passed']}/"
                f"{baseline_summary['num_tasks']} "
                f"({baseline_summary['pass_rate']*100:.2f}% pass rate)"
            )

            print(f"{benchmark} on {provider}:{model_name} | SELF-DEBUG")
            selfdbg_summary = evaluate_benchmark_on_model(
                benchmark=benchmark,
                provider=provider,
                model_name=model_name,
                max_tasks=max_tasks,
                mode="self_debug",
                max_self_debug_iters=max_self_debug_iters,
            )

            print(
                f"Self-debug: {selfdbg_summary['num_passed']}/"
                f"{selfdbg_summary['num_tasks']} "
                f"({selfdbg_summary['pass_rate']*100:.2f}% pass rate)"

            )

            delta_pass = selfdbg_summary["num_passed"] - baseline_summary["num_passed"]
            delta_rate = (selfdbg_summary["pass_rate"] - baseline_summary["pass_rate"]) * 100.0

            print(
                f"Improvement: +{delta_pass} tasks, "
                f"{delta_rate:+.4f} percentage points"
            )

            all_results.append(("baseline", baseline_summary))
            all_results.append(("self_debug", selfdbg_summary))



    # Bar Chart: Baseline vs Self-Debug pass rates
    combined: Dict[tuple, Dict[str, float]] = {}
    for mode, summary in all_results:
        key = (summary["benchmark"], summary["provider"], summary["model"])
        if key not in combined:
            combined[key] = {}
        combined[key][mode] = summary["pass_rate"] * 100.0

    labels = []
    baseline_rates = []
    selfdebug_rates = []

    for (benchmark, provider, model), modes in combined.items():
        if "baseline" in modes and "self_debug" in modes:
            labels.append(f"{benchmark}\n{provider}:{model}")
            baseline_rates.append(modes["baseline"])
            selfdebug_rates.append(modes["self_debug"])

    if labels:
        x = range(len(labels))
        width = 0.4

        plt.figure(figsize=(12, 6))
        plt.bar([i - width / 2 for i in x], baseline_rates, width, label="Baseline")
        plt.bar([i + width / 2 for i in x], selfdebug_rates, width, label="Self-Debug")

        plt.xticks(list(x), labels, rotation=30, ha="right")
        plt.ylabel("Pass Rate (%)")
        plt.title("Baseline vs Self-Debugging Pass Rates\n(per Benchmark & Model)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("self_debug_comparison.png")
        plt.show()


    print("\nSAMPLE ORIGINAL VS CORRECTED CODE (ONLY SUCCESSFUL SELF-DEBUG CASES)\n")

    sample_count = 0
    max_samples = 2 

    for mode, summary in all_results:
        if sample_count >= max_samples:
            break

        if mode != "self_debug":
            continue

        details = summary.get("details", [])
        if not details:
            continue

        for d in details:
            if sample_count >= max_samples:
                break

            # We only want:
            # - self-debug was actually used
            # - initial code failed
            # - final code passed
            # - there was at least one patch (num_iterations > 1)
            used_self_debug = d.get("self_debug_used", False)
            final_passed = d.get("passed", False)
            num_iters = d.get("num_iterations", 1)
            init_num_passed = d.get("initial_num_passed", None)
            init_num_tests = d.get("initial_num_tests", None)

            if not used_self_debug:
                continue
            if not final_passed:
                continue
            if num_iters <= 1:
                continue
            # initial must have failed: either 0 passed or < total tests
            if init_num_tests is not None and init_num_passed is not None:
                if init_num_passed == init_num_tests:
                    # initial already passed all tests; skip
                    continue

            init_code = d.get("initial_code", "") or ""
            final_code = d.get("final_code", "") or ""
            patch_explanations = d.get("patch_explanations", []) or []
            prompt_to_model = d.get("prompt_to_model", "") or ""
            initial_explanation = d.get("initial_explanation", "") or ""
            initial_error_type = d.get("initial_error_type", "") or ""
            initial_error_message = d.get("initial_error_message", "") or ""

            print("============================================================")
            print(f"Benchmark: {summary['benchmark']}")
            print(f"Model: {summary['provider']}:{summary['model']}")
            print(f"Task ID: {d.get('task_id')}")
            print(f"Initial tests: {init_num_passed}/{init_num_tests}")
            print(f"Final tests: {d.get('num_passed')}/{d.get('num_tests')}")
            print(f"Iterations used: {num_iters}")
            print("============================================================\n")

            # 1) Raw prompt sent to the model
            print(">>> RAW PROMPT SENT TO MODEL (TRUNCATED) <<<\n")
            # You can truncate if it's too long
            print(prompt_to_model[:2000])
            if len(prompt_to_model) > 2000:
                print("\n...[prompt truncated]...\n")

            print("\n>>> INITIAL CODE (FAILED) <<<\n")
            print(init_code)

            print("\n>>> INITIAL CODE EXPLANATION <<<\n")
            print(initial_explanation)
      
            print("\n>>> INITIAL ERROR <<<\n")
            print(f"Error type: {initial_error_type}")
            print(f"Error message: {initial_error_message}")

            print("\n>>> CORRECTED CODE (ALL TESTS PASSED) <<<\n")
            print(final_code)
    
            if patch_explanations:
                print("\n>>> PATCH EXPLANATION(S) <<<")
                for i, expl in enumerate(patch_explanations, 1):
                    print(f"\n[Patch {i}]\n{expl}")

            sample_count += 1

