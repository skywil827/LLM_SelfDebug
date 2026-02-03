
import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, List, Literal
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
from datasets import load_dataset
from human_eval.data import read_problems
import matplotlib.pyplot as plt
from humanEvalInput import HumanEvalTask, load_humaneval_task
from MBPPInput import MBPPTask, load_mbpp_task
from APPSInput import APPSTask, load_apps_task
from sweBenchInput import load_swe_instance, build_swe_prompt, SWELITETask
from runtime.feedback_package import build_feedback_package
from runtime.code_exec import ExecutionResult, execute_task
from error_explanation import (
    build_error_explanation_io,
    diagnose_bug_with_openai,
    build_error_explanation_text,
)
from patch_generation import produce_next_code_version



load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

openai_client = OpenAI()
gemini_client = genai.Client(api_key=google_api_key)

# Safe defaults for models
OPENAI_MODEL = "gpt-4o-mini"
GOOGLE_MODEL = "gemini-2.0-flash"


TaskType = Union[HumanEvalTask, MBPPTask, APPSTask, SWELITETask]
Provider = Literal["openai", "gemini"]


@dataclass(frozen=True)
class AgentSpec:
    provider: Provider
    model: str


def pick_fixer(patch_agents: List["AgentSpec"], iteration: int) -> "AgentSpec":
    """
    Sequential handoff:
    iteration=1 -> patch_agents[0]
    iteration=2 -> patch_agents[1]
    ...
    wraps around if iterations > len(patch_agents)
    """
    if not patch_agents:
        raise ValueError("patch_agents cannot be empty for sequential handoff.")
    return patch_agents[(iteration - 1) % len(patch_agents)]


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


def _get_task_identity(task: TaskType) -> tuple[str, str]:
    benchmark = task.constraints.get("benchmark", "UNKNOWN")

    if isinstance(task, HumanEvalTask):
        tid = task.task_id
    elif isinstance(task, MBPPTask):
        tid = f"MBPP/{task.task_id}"
    elif isinstance(task, APPSTask):
        tid = f"APPS/{task.problem_id}"
    elif isinstance(task, SWELITETask):
        tid = f"SWELITE/{task.instance_id}"
    else:
        tid = "UNKNOWN"

    return benchmark, tid


# ------------------------------
# Initial generation (OpenAI)
# ------------------------------
def generate_initial_code_with_openai(
    task: TaskType,
    model: str = OPENAI_MODEL,
) -> InitialCodeResult:
    benchmark, tid = _get_task_identity(task)

    if isinstance(task, SWELITETask):
        base_prompt = build_swe_prompt(task)
        response_schema = """
        {
          "plan": "Short step-by-step reasoning about how you will solve the problem.",
          "patch": "Unified diff patch to apply on top of the base commit. Do NOT wrap in ``` fences.",
          "explanation": "A clear explanation of how the patch fixes the bug."
        }
        """
        artifact_key = "patch"
        constraints_block = """
        Constraints:
        - The "patch" MUST be a valid unified diff starting with: diff --git a/... b/...
        - Do NOT include backticks or Markdown fences in any field.
        - Return ONLY code changes in the patch; do not include prose outside the JSON.
        """
    else:
        base_prompt = task.build_prompt()
        response_schema = """
        {
          "plan": "Short step-by-step reasoning about how you will solve the problem.",
          "code": "Python code implementing your solution. Do NOT wrap in ``` fences.",
          "explanation": "A clear explanation of how the code works."
        }
        """
        artifact_key = "code"
        constraints_block = """
        Constraints:
        - The "code" MUST be valid Python.
        - Do NOT include backticks or Markdown fences in any field.
        - For HumanEval/MBPP, implement ONLY the required function(s), not a CLI.
        - For APPS, you may write a full program if needed, but keep it minimal and correct.
        """

    system_msg = (
        "You are a highly reliable coding assistant. "
        "Your job is to understand the problem, propose a clear plan, "
        "then produce the required artifact, and finally explain your solution."
    )

    user_instructions = f"""
        You will receive a programming task from the {benchmark} benchmark.

        TASK SPECIFICATION
        ------------------
        {base_prompt}

        RESPONSE FORMAT (MANDATORY)
        ---------------------------
        Respond ONLY as a JSON object with the following fields:

        {response_schema}

        {constraints_block}
    """

    response = openai_client.chat.completions.create(
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
        parsed = {"plan": "", artifact_key: "", "explanation": content}

    plan = str(parsed.get("plan", "") or "").strip()
    artifact = str(parsed.get(artifact_key, "") or "").strip()
    explanation = str(parsed.get("explanation", "") or "").strip()

    return InitialCodeResult(
        benchmark=benchmark,
        task_id=tid,
        model=model,
        plan=plan,
        code=artifact,
        explanation=explanation,
        raw_response=response.model_dump() if hasattr(response, "model_dump") else {"raw": str(response)},
        raw_prompt=user_instructions,
    )



# Initial generation (Gemini)
def generate_initial_code_with_gemini(
    task: TaskType,
    model: str = GOOGLE_MODEL,
) -> InitialCodeResult:
    benchmark, tid = _get_task_identity(task)

    if isinstance(task, SWELITETask):
        base_prompt = build_swe_prompt(task)
        expected_key = "patch"
        output_hint = (
            'Return ONLY JSON with keys: {"plan","patch","explanation"}. '
            '"patch" must be a valid unified diff starting with diff --git.'
        )
    else:
        base_prompt = task.build_prompt()
        expected_key = "code"
        output_hint = (
            'Return ONLY JSON with keys: {"plan","code","explanation"}. '
            '"code" must be valid Python and must not be wrapped in backticks.'
        )

    system_msg = (
        "You are a highly reliable coding assistant. "
        "You must follow the user's output schema exactly. "
        "Do not include Markdown fences. Return strict JSON only."
    )

    user_instructions = f"""
        You will receive a task from the {benchmark} benchmark.

        SYSTEM INSTRUCTIONS
        {system_msg}

        TASK SPECIFICATION
        ------------------
        {base_prompt}

        RESPONSE FORMAT (MANDATORY)
        ---------------------------
        {output_hint}

        General rules:
        - Output MUST be a single JSON object (no extra text).
        - Do NOT include backticks or Markdown fences in any field.
        - Keep the plan short and concrete.
        """.strip()

    response = gemini_client.models.generate_content(
        model=model,
        contents=types.Part.from_text(text=user_instructions),
        config=types.GenerateContentConfig(temperature=0.3),
    )

    content = getattr(response, "text", "") or str(response)
    content = content.strip()

    if content.startswith("```"):
        lines = content.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"plan": "", expected_key: "", "explanation": content}

    plan = str(parsed.get("plan", "") or "").strip()
    explanation = str(parsed.get("explanation", "") or "").strip()
    artifact = str(parsed.get(expected_key, "") or "").strip()

    raw_response = response.to_dict() if hasattr(response, "to_dict") else {"raw": str(response)}

    return InitialCodeResult(
        benchmark=benchmark,
        task_id=tid,
        model=model,
        plan=plan,
        code=artifact,
        explanation=explanation,
        raw_response=raw_response,
        raw_prompt=user_instructions,
    )


def generate_initial(task: TaskType, provider: str, model_name: str) -> InitialCodeResult:
    if provider == "openai":
        return generate_initial_code_with_openai(task, model=model_name)
    if provider == "gemini":
        return generate_initial_code_with_gemini(task, model=model_name)
    raise ValueError(f"Unknown provider: {provider}")


# Self-debug FROM a baseline candidate
def self_debug_from_candidate(
    task: TaskType,
    initial_code: str,
    first_exec: ExecutionResult,
    max_self_debug_iters: int = 3,
    patch_model: str = "gpt-4o",
) -> Dict[str, Any]:
    if first_exec.passed or first_exec.num_tests == 0:
        return {
            "benchmark": first_exec.benchmark,
            "task_id": first_exec.task_id,
            "mode": "self_debug",
            "passed": first_exec.passed,
            "num_tests": first_exec.num_tests,
            "num_passed": first_exec.num_passed,
            "error_type": first_exec.error_type,
            "error_message": first_exec.error_message,
            "num_iterations": 1,
            "self_debug_used": False,
            "patch_models_used": [],
            "initial_code": initial_code,
            "final_code": initial_code,
            "patch_explanations": [],
            "initial_error_type": first_exec.error_type,
            "initial_error_message": first_exec.error_message,
            "initial_num_tests": first_exec.num_tests,
            "initial_num_passed": first_exec.num_passed,
        }

    current = initial_code
    final_exec_result: Optional[ExecutionResult] = first_exec
    num_iterations = 1

    patch_explanations: list[str] = []
    patch_models_used: list[str] = []

    for it in range(1, max_self_debug_iters + 1):
        num_iterations = it + 1

        feedback = build_feedback_package(task, current, final_exec_result)
        io_bundle = build_error_explanation_io(task, current, feedback)

        diag = diagnose_bug_with_openai(io_bundle)
        err_expl = build_error_explanation_text(io_bundle, diag)

        next_candidate, patch_info = produce_next_code_version(
            task, current, err_expl, model=patch_model
        )
        current = next_candidate

        patch_models_used.append(f"openai:{patch_model}")
        patch_explanations.append(getattr(patch_info, "rationale", None) or "")

        exec_result = execute_task(task, current)
        final_exec_result = exec_result

        if exec_result.passed or exec_result.num_tests == 0:
            break

    return {
        "benchmark": final_exec_result.benchmark,
        "task_id": final_exec_result.task_id,
        "mode": "self_debug",
        "passed": final_exec_result.passed,
        "num_tests": final_exec_result.num_tests,
        "num_passed": final_exec_result.num_passed,
        "error_type": final_exec_result.error_type,
        "error_message": final_exec_result.error_message,
        "num_iterations": num_iterations,
        "self_debug_used": True,
        "patch_models_used": patch_models_used,
        "initial_code": initial_code,
        "final_code": current,
        "patch_explanations": patch_explanations,
        "initial_error_type": first_exec.error_type,
        "initial_error_message": first_exec.error_message,
        "initial_num_tests": first_exec.num_tests,
        "initial_num_passed": first_exec.num_passed,
    }


def sequential_handoff_from_candidate(
    task: TaskType,
    initial_code: str,
    first_exec: ExecutionResult,
    patch_agents: List[AgentSpec],
    max_self_debug_iters: int = 3,
) -> Dict[str, Any]:
    if first_exec.passed or first_exec.num_tests == 0:
        return {
            "benchmark": first_exec.benchmark,
            "task_id": first_exec.task_id,
            "mode": "sequential_handoff",
            "passed": first_exec.passed,
            "num_tests": first_exec.num_tests,
            "num_passed": first_exec.num_passed,
            "error_type": first_exec.error_type,
            "error_message": first_exec.error_message,
            "num_iterations": 1,
            "self_debug_used": False,
            "patch_models_used": [],
            "initial_code": initial_code,
            "final_code": initial_code,
            "patch_explanations": [],
        }

    current = initial_code
    final_exec_result: Optional[ExecutionResult] = first_exec
    num_iterations = 1

    patch_explanations: list[str] = []
    patch_models_used: list[str] = []

    for it in range(1, max_self_debug_iters + 1):
        num_iterations = it + 1

        feedback = build_feedback_package(task, current, final_exec_result)
        io_bundle = build_error_explanation_io(task, current, feedback)

        diag = diagnose_bug_with_openai(io_bundle)
        err_expl = build_error_explanation_text(io_bundle, diag)

        fixer = pick_fixer(patch_agents, it)
        if fixer.provider != "openai":
            raise ValueError("This setup currently supports OpenAI patch agents only.")

        next_candidate, patch_info = produce_next_code_version(
            task, current, err_expl, model=fixer.model
        )
        current = next_candidate

        patch_models_used.append(f"{fixer.provider}:{fixer.model}")
        patch_explanations.append(getattr(patch_info, "rationale", None) or "")

        exec_result = execute_task(task, current)
        final_exec_result = exec_result

        if exec_result.passed or exec_result.num_tests == 0:
            break

    return {
        "benchmark": final_exec_result.benchmark,
        "task_id": final_exec_result.task_id,
        "mode": "sequential_handoff",
        "passed": final_exec_result.passed,
        "num_tests": final_exec_result.num_tests,
        "num_passed": final_exec_result.num_passed,
        "error_type": final_exec_result.error_type,
        "error_message": final_exec_result.error_message,
        "num_iterations": num_iterations,
        "self_debug_used": True,
        "patch_models_used": patch_models_used,
        "initial_code": initial_code,
        "final_code": current,
        "patch_explanations": patch_explanations,
    }


# ------------------------------
# Plotting (presentable, headless-safe)
# ------------------------------
def shorten(s: str, max_len: int = 26) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def add_value_labels(ax, bars):
    for b in bars:
        h = float(b.get_height() or 0.0)
        ax.annotate(
            f"{h:.1f}%",
            (b.get_x() + b.get_width() / 2, h),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_clean_grouped_bars(all_results: List[tuple], k_values: List[int]) -> None:
    by_benchmark = defaultdict(list)
    for mode_tag, summary in all_results:
        by_benchmark[summary["benchmark"]].append((mode_tag, summary))

    series_order = ["baseline", "self_debug_single"] + [f"handoff_{k}agents" for k in k_values]

    for benchmark, entries in by_benchmark.items():
        grouped = defaultdict(dict)
        for mode_tag, summary in entries:
            key = (summary["provider"], summary["model"])
            grouped[key][mode_tag] = summary["pass_rate"] * 100.0

        keys = list(grouped.keys())
        labels = [shorten(f"{p}:{m}", 28) for (p, m) in keys]
        series_vals = {s: [grouped[k].get(s, 0.0) for k in keys] for s in series_order}

        x = list(range(len(keys)))
        width = 0.12
        mid = (len(series_order) - 1) / 2.0
        offsets = [(i - mid) * width for i in range(len(series_order))]

        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.6)

        for i, s in enumerate(series_order):
            bars = ax.bar([xi + offsets[i] for xi in x], series_vals[s], width, label=s)
            add_value_labels(ax, bars)

        ax.set_title(f"{benchmark}: Baseline vs Single Self-Debug vs Sequential Handoff")
        ax.set_ylabel("Pass Rate (%)")
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

        fig.tight_layout()
        out = f"pass_rates_{benchmark}_clean.png".replace("/", "_")
        plt.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_improvement_over_baseline(all_results: List[tuple], k_values: List[int]) -> None:
    baseline_lookup: Dict[tuple, float] = {}
    rates_lookup: Dict[tuple, Dict[str, float]] = defaultdict(dict)

    for mode_tag, summary in all_results:
        key = (summary["benchmark"], summary["provider"], summary["model"])
        rate = summary["pass_rate"] * 100.0
        rates_lookup[key][mode_tag] = rate
        if mode_tag == "baseline":
            baseline_lookup[key] = rate

    benchmarks_in_results = sorted({k[0] for k in rates_lookup.keys()})

    for bench in benchmarks_in_results:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.6)

        for (b, provider, model), modes in rates_lookup.items():
            if b != bench:
                continue
            base = baseline_lookup.get((b, provider, model), 0.0)

            xs = [1]
            ys = [modes.get("self_debug_single", 0.0) - base]

            for k in k_values:
                tag = f"handoff_{k}agents"
                xs.append(k)
                ys.append(modes.get(tag, 0.0) - base)

            label = shorten(f"{provider}:{model}", 35)
            ax.plot(xs, ys, marker="o", label=label)

        ax.axhline(0, linewidth=1)
        ax.set_title(f"{bench}: Improvement over Baseline vs # Patch Agents (Sequential Handoff)")
        ax.set_xlabel("Number of patch agents (K)")
        ax.set_ylabel("Δ Pass Rate (percentage points)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

        fig.tight_layout()
        out = f"improvement_vs_k_{bench}.png".replace("/", "_")
        plt.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)


# ------------------------------
# Utilities: task loading + summaries
# ------------------------------
def load_tasks_for_benchmark(benchmark: str, max_tasks: Optional[int]) -> List[TaskType]:
    if benchmark == "HumanEval":
        problems = read_problems()
        task_ids = list(problems.keys())
        if max_tasks is not None:
            task_ids = task_ids[:max_tasks]
        return [load_humaneval_task(tid) for tid in task_ids]

    if benchmark == "MBPP":
        ds = load_dataset("mbpp", "sanitized", split="test")
        if max_tasks is not None:
            ds = ds.select(range(max_tasks))
        return [load_mbpp_task(int(row["task_id"])) for row in ds]

    if benchmark == "APPS":
        ds = load_dataset("codeparrot/apps", split="test")
        if max_tasks is not None:
            ds = ds.select(range(max_tasks))
        return [load_apps_task(int(row["problem_id"]), split="test") for row in ds]

    if benchmark == "SWE-bench_LITE":
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        if max_tasks is not None:
            ds = ds.select(range(max_tasks))
        return [load_swe_instance(row["instance_id"]) for row in ds]

    raise ValueError(f"Unknown benchmark: {benchmark}")


def summarize_results(details: List[Dict[str, Any]], benchmark: str, provider: str, model: str, mode: str) -> Dict[str, Any]:
    total = len(details)
    passed = sum(1 for d in details if d.get("passed"))
    return {
        "benchmark": benchmark,
        "provider": provider,
        "model": model,
        "mode": mode,
        "num_tasks": total,
        "num_passed": passed,
        "pass_rate": (passed / total) if total else 0.0,
        "details": details,
    }


# ------------------------------
# Main experiment (baseline once + conditional self-debug)
# ------------------------------
if __name__ == "__main__":
    configs = [
        ("gemini", "gemini-2.5-pro"),
        ("openai", "gpt-4o-mini"),
    ]
    benchmarks = ["HumanEval"]
    # benchmarks = ["HumanEval", "MBPP", "APPS", "SWE-bench_LITE"]
    max_tasks = 5
    max_self_debug_iters = 2

    single_patch_model = "gpt-4o"

    patch_pool: List[AgentSpec] = [
        AgentSpec("openai", "gpt-4o-mini"),
        AgentSpec("openai", "gpt-4o"),
        AgentSpec("openai", "gpt-4.1-mini"),
        AgentSpec("openai", "gpt-4.1"),
    ]

    k_values = [2, 3, 4]  # e.g. [2, 3, 4]

    all_results: List[tuple[str, Dict[str, Any]]] = []


    for provider, model_name in configs:
        for benchmark in benchmarks:
            tasks = load_tasks_for_benchmark(benchmark, max_tasks)

            baseline_details: List[Dict[str, Any]] = []
            single_details: List[Dict[str, Any]] = []
            handoff_details_by_k: Dict[int, List[Dict[str, Any]]] = {k: [] for k in k_values}

            for task in tasks:
                # 1) generate once
                init = generate_initial(task, provider, model_name)
                candidate = init.code

                # 2) baseline once
                base_exec = execute_task(task, candidate)
                baseline_details.append(
                    {
                        "benchmark": base_exec.benchmark,
                        "task_id": base_exec.task_id,
                        "provider": provider,
                        "model": model_name,
                        "mode": "baseline",
                        "passed": base_exec.passed,
                        "num_tests": base_exec.num_tests,
                        "num_passed": base_exec.num_passed,
                        "error_type": base_exec.error_type,
                        "error_message": base_exec.error_message,
                        "num_iterations": 1,
                        "self_debug_used": False,
                        "initial_code": candidate,
                        "final_code": candidate,
                    }
                )

                # 3) self-debug ONLY if baseline failed 
                single_res = self_debug_from_candidate(
                    task=task,
                    initial_code=candidate,
                    first_exec=base_exec,
                    max_self_debug_iters=max_self_debug_iters,
                    patch_model=single_patch_model,
                )
                single_res.update({"provider": provider, "model": model_name})
                single_details.append(single_res)

                # 4) sequential handoff ONLY if baseline failed
                for k in k_values:
                    agents_k = patch_pool[:k]
                    handoff_res = sequential_handoff_from_candidate(
                        task=task,
                        initial_code=candidate,
                        first_exec=base_exec,
                        patch_agents=agents_k,
                        max_self_debug_iters=max_self_debug_iters,
                    )
                    handoff_res.update({"provider": provider, "model": model_name})
                    handoff_details_by_k[k].append(handoff_res)

            baseline_summary = summarize_results(baseline_details, benchmark, provider, model_name, "baseline")
            single_summary = summarize_results(single_details, benchmark, provider, model_name, "self_debug_single")

            all_results.append(("baseline", baseline_summary))
            all_results.append(("self_debug_single", single_summary))

            for k in k_values:
                tag = f"handoff_{k}agents"
                summary_k = summarize_results(handoff_details_by_k[k], benchmark, provider, model_name, tag)
                all_results.append((tag, summary_k))

            print(f"\n{benchmark} on {provider}:{model_name}")
            print(f"Baseline: {baseline_summary['num_passed']}/{baseline_summary['num_tasks']} ({baseline_summary['pass_rate']*100:.2f}%)")
            print(f"Self-debug (only on failures): {single_summary['num_passed']}/{single_summary['num_tasks']} ({single_summary['pass_rate']*100:.2f}%)")
            for k in k_values:
                tag = f"handoff_{k}agents"
                summary_k = next(s for t, s in all_results if t == tag and s["benchmark"] == benchmark and s["provider"] == provider and s["model"] == model_name)
                print(f"Handoff ({k}): {summary_k['num_passed']}/{summary_k['num_tasks']} ({summary_k['pass_rate']*100:.2f}%)")

    # plots (both)
    plot_clean_grouped_bars(all_results, k_values)
    plot_improvement_over_baseline(all_results, k_values)
