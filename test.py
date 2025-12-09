import os
import io
import json
import traceback
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




# Load environment variables from .env
load_dotenv(override=True)

# Explicitly set environment variables for APIs
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize API clients
openai = OpenAI()
google.generativeai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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
    """
    Initial Understanding & First Code stage using OpenAI (e.g., gpt-4o, gpt-4o-mini).
    """
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
    )



# Initial Code Generation – Gemini
def generate_initial_code_with_gemini(
    task: TaskType,
    model: str = GOOGLE_MODEL,
) -> InitialCodeResult:
    """
    Initial Understanding & First Code stage using Gemini (e.g., gemini-2.0-flash).
    Mirrors generate_initial_code_with_openai, but uses Gemini APIs.
    """
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
    )


# Baseline (NO self-debugging)

def run_single_task_no_self_debug(
    task: TaskType,
    provider: str,
    model_name: str,
) -> Dict[str, Any]:
    """
    Baseline pipeline: single shot code generation + execution, no self-debug loop.
    """
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
    }


# Self-Debugging – Single Task
def run_single_task_with_self_debug(
    task: TaskType,
    provider: str,
    model_name: str,
    max_self_debug_iters: int = 3,
) -> Dict[str, Any]:
    """
    Full self-debugging pipeline for a single task with a given provider/model.
    """
    if provider == "openai":
        init = generate_initial_code_with_openai(task, model=model_name)
    elif provider == "gemini":
        init = generate_initial_code_with_gemini(task, model=model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    current_code = init.code
    num_iterations = 0
    used_self_debug = False
    final_exec_result: Optional[ExecutionResult] = None

    for it in range(max_self_debug_iters + 1):
        num_iterations = it + 1

     
        exec_result = execute_task_code(task, current_code)
        final_exec_result = exec_result

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
    }



def evaluate_benchmark_on_model(
    benchmark: str,
    provider: str,
    model_name: str,
    max_tasks: Optional[int] = None,
    mode: str = "baseline",
    max_self_debug_iters: int = 3,
) -> Dict[str, Any]:
    """
    Run ALL (or first max_tasks) tasks of a benchmark for a given provider/model.

    mode ∈ {"baseline", "self_debug"}
    """
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


    max_tasks = 1
    max_self_debug_iters = 3

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

    # print("\nOVERALL COMPARISON SUMMARY")
    # for mode, s in all_results:
    #     label = "BASELINE" if mode == "baseline" else "SELF-DEBUG"
    #     print(
    #         f"{label} | {s['benchmark']} | {s['provider']}:{s['model']} "
    #         f"-> {s['num_passed']}/{s['num_tasks']} passed "
    #         f"({s['pass_rate']*100:.2f}%)"
    #     )

   
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
        plt.bar([i - width/2 for i in x], baseline_rates, width, label="Baseline")
        plt.bar([i + width/2 for i in x], selfdebug_rates, width, label="Self-Debug")

        plt.xticks(list(x), labels, rotation=30, ha="right")
        plt.ylabel("Pass Rate (%)")
        plt.title("Baseline vs Self-Debugging Pass Rates\n(per Benchmark & Model)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("self_debug_comparison.png")
        plt.show()
    else:
        print("No paired baseline/self-debug results available to plot.")



  
    # Debugging Attempts (Self-Debug Mode Only)
    # print("\n===== SELF-DEBUGGING ATTEMPTS REPORT =====")

    # attempts_summary: Dict[tuple, Dict[str, Any]] = {}
    # attempts_dist_by_benchmark: Dict[str, Dict[int, int]] = {}

    # for mode, summary in all_results:
    #     if mode != "self_debug":
    #         continue

    #     key = (summary["benchmark"], summary["provider"], summary["model"])
    #     details = summary.get("details", [])

    #     num_tasks = summary["num_tasks"]
    #     iterations = [d.get("num_iterations", 1) for d in details]
    #     used_flags = [d.get("self_debug_used", False) for d in details]

    #     total_iters = sum(iterations) if iterations else 0
    #     avg_iters_all = total_iters / num_tasks if num_tasks > 0 else 0.0

    #     used_count = sum(1 for u in used_flags if u)
    #     iters_when_used = [it for it, u in zip(iterations, used_flags) if u]
    #     avg_iters_used = (
    #         (sum(iters_when_used) / len(iters_when_used)) if iters_when_used else 0.0
    #     )
    #     max_iters = max(iterations) if iterations else 0

    #     attempts_summary[key] = {
    #         "num_tasks": num_tasks,
    #         "used_count": used_count,
    #         "avg_iters_all": avg_iters_all,
    #         "avg_iters_used": avg_iters_used,
    #         "max_iters": max_iters,
    #     }


    #     bm = summary["benchmark"]
    #     if bm not in attempts_dist_by_benchmark:
    #         attempts_dist_by_benchmark[bm] = {}

    #     for it in iterations:
    #         attempts = max(0, it - 1)
    #         attempts_dist_by_benchmark[bm][attempts] = (
    #             attempts_dist_by_benchmark[bm].get(attempts, 0) + 1
    #         )

    # if not attempts_summary:
    #     print("No self-debugging data available.")
    # else:
    #     for (benchmark, provider, model), stats in attempts_summary.items():
    #         print(
    #             f"\n[Self-Debug Attempts] {benchmark} | {provider}:{model}\n"
    #             f"- Tasks evaluated: {stats['num_tasks']}\n"
    #             f"- Tasks where self-debug was actually used: {stats['used_count']}\n"
    #             f"- Avg iterations per task (including ones that passed immediately): "
    #             f"{stats['avg_iters_all']:.2f}\n"
    #             f"- Avg iterations only on tasks that used self-debug: "
    #             f"{stats['avg_iters_used']:.2f}\n"
    #             f"- Max iterations used on any single task: {stats['max_iters']}"
    #         )


    # Pie Charts
    # if attempts_dist_by_benchmark:
    #     benchmarks_list = sorted(attempts_dist_by_benchmark.keys())
    #     n = len(benchmarks_list)

    #     fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    #     if n == 1:
    #         axes = [axes]

    #     for ax, bm in zip(axes, benchmarks_list):
    #         dist = attempts_dist_by_benchmark[bm]
    #         attempt_values = sorted(dist.keys())
    #         counts = [dist[a] for a in attempt_values]
    #         total = sum(counts)

    #         if total == 0:
    #             ax.text(0.5, 0.5, "No tasks", ha="center", va="center")
    #             ax.set_axis_off()
    #             continue

    #         wedges, _ = ax.pie(
    #             counts,
    #             startangle=90,
    #         )
    #         ax.set_title(bm)


    #         legend_labels = [
    #             f"{a}: {count / total * 100:.1f}%"
    #             for a, count in zip(attempt_values, counts)
    #         ]
    #         ax.legend(
    #             wedges,
    #             legend_labels,
    #             title="Debugging Attempts",
    #             loc="center left",
    #             bbox_to_anchor=(1, 0.5),
    #         )
    #         ax.axis("equal")

    #     plt.tight_layout()
    #     plt.savefig("self_debug_attempt_distribution_per_benchmark.png")
    #     plt.show()
    # else:
    #     print("No self-debug attempt distributions available for pie charts.")
