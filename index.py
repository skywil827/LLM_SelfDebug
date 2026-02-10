import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, List, Literal, Tuple
from collections import defaultdict
from pathlib import Path
from datetime import datetime
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



def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def make_run_dir(out_root: str = "results") -> Tuple[str, str, str]:
    """
    Creates:
      results/run_<timestamp>/
        - results.json
        - plots/
    Returns: (timestamp, run_dir, plots_dir)
    """
    ts = _now_ts()
    run_dir = Path(out_root) / f"run_{ts}"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return ts, str(run_dir), str(plots_dir)


def build_summary_report_text(compact_summary_rows: List[Dict[str, Any]]) -> List[str]:
    """
    Builds a human-readable summary identical to terminal output,
    suitable for inclusion at the end of results.json.
    """
    lines: List[str] = []

    for row in compact_summary_rows:
        bench = row["benchmark"]
        provider = row["provider"]
        model = row["model"]

        bsum = row["baseline"]
        ssum = row["self_debug"]

        lines.append(f"{bench} on {provider}:{model}")
        lines.append(
            f"Baseline: {bsum['num_passed']}/{bsum['num_tasks']} "
            f"({bsum['pass_rate']*100:.2f}%)"
        )
        lines.append(
            f"Self-debug (only on failures): {ssum['num_passed']}/{ssum['num_tasks']} "
            f"({ssum['pass_rate']*100:.2f}%)"
        )

        for k, hsum in row["handoff_by_k"].items():
            lines.append(
                f"Handoff ({k}): {hsum['num_passed']}/{hsum['num_tasks']} "
                f"({hsum['pass_rate']*100:.2f}%)"
            )

        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()

    return lines


def save_experiment_results(
    *,
    run_dir: str,
    timestamp: str,
    summaries: List[Tuple[str, Dict[str, Any]]],
    details: Dict[str, Any],
    config: Dict[str, Any],
    artifacts: Dict[str, Any],
) -> str:
    out_path = Path(run_dir) / "results.json"
    payload = {
        "timestamp": timestamp,
        "config": config,
        "summaries": [{"mode_tag": mode_tag, **summary} for mode_tag, summary in summaries],
        "details": details,
        "artifacts": artifacts,
        "summary_report_text": artifacts.get("summary_report_text"),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[LOG] Results saved to {out_path}")
    return str(out_path)


load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

openai_client = OpenAI()
gemini_client = genai.Client(api_key=google_api_key)

OPENAI_MODEL = "gpt-4o"
GOOGLE_MODEL = "gemini-2.0-flash"

TaskType = Union[HumanEvalTask, MBPPTask, APPSTask, SWELITETask]
Provider = Literal["openai", "gemini"]


@dataclass(frozen=True)
class AgentSpec:
    provider: Provider
    model: str


def pick_fixer(patch_agents: List["AgentSpec"], iteration: int) -> "AgentSpec":
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


def _get_task_identity(task) -> tuple[str, str]:
    constraints = getattr(task, "constraints", {}) or {}
    benchmark = constraints.get("benchmark", "UNKNOWN")

    if hasattr(task, "task_id") and getattr(task, "task_id") is not None:
        tid = str(getattr(task, "task_id"))
        if benchmark == "MBPP" and not tid.startswith("MBPP/"):
            tid = f"MBPP/{tid}"
        return benchmark, tid

    if hasattr(task, "problem_id") and getattr(task, "problem_id") is not None:
        return benchmark, f"APPS/{getattr(task, 'problem_id')}"

    if hasattr(task, "instance_id") and getattr(task, "instance_id") is not None:
        return benchmark, f"SWELITE/{getattr(task, 'instance_id')}"

    return benchmark, "UNKNOWN"



# Initial generation (OpenAI / Gemini)
def generate_initial_code_with_openai(task: TaskType, model: str = OPENAI_MODEL) -> InitialCodeResult:
    benchmark, tid = _get_task_identity(task)

    if isinstance(task, SWELITETask):
        base_prompt = build_swe_prompt(task)
        artifact_key = "patch"

        response_schema = """
        {
          "plan": "...",
          "patch": "...",
          "explanation": "..."
        }
        """
        constraints_block = """
        Constraints:
        - The "patch" MUST be a valid unified diff starting with: diff --git a/... b/...
        - Do NOT include backticks or Markdown fences.
        """
    else:
        base_prompt = task.build_prompt()
        artifact_key = "code"
        response_schema = """
        {
          "plan": "...",
          "code": "...",
          "explanation": "..."
        }
        """
        constraints_block = """
        Constraints:
        - The "code" MUST be valid Python.
        - Do NOT include backticks or Markdown fences.
        """

    system_msg = (
        "You are a highly reliable coding assistant. "
        "Follow the schema and return JSON only."
    )

    user_instructions = f"""
        TASK ({benchmark})
        ------------------
        {base_prompt}

        Return JSON:
        {response_schema}

        """.strip()

    response = openai_client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_instructions},
        ],
    )

    content = response.choices[0].message.content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"plan": "", artifact_key: "", "explanation": content}

    return InitialCodeResult(
        benchmark=benchmark,
        task_id=tid,
        model=model,
        plan=str(parsed.get("plan", "") or "").strip(),
        code=str(parsed.get(artifact_key, "") or "").strip(),
        explanation=str(parsed.get("explanation", "") or "").strip(),
        raw_response=response.model_dump() if hasattr(response, "model_dump") else {"raw": str(response)},
        raw_prompt=user_instructions,
    )


def generate_initial_code_with_gemini(task: TaskType, model: str = GOOGLE_MODEL) -> InitialCodeResult:
    benchmark, tid = _get_task_identity(task)
    # expected_key = ""

    if isinstance(task, SWELITETask):
        base_prompt = build_swe_prompt(task)
        expected_key = "patch"

        response_schema = """
        {
          "plan": "...",
          "patch": "...",
          "explanation": "..."
        }
        """
        constraints_block = """
        Constraints:
        - The "patch" MUST be a valid unified diff starting with: diff --git a/... b/...
        - Do NOT include backticks or Markdown fences.
        """
    else:
        base_prompt = task.build_prompt()
        expected_key = "code"

        response_schema = """
        {
          "plan": "...",
          "code": "...",
          "explanation": "..."
        }
        """
        constraints_block = """
        Constraints:
        - The "code" MUST be valid Python.
        - Do NOT include backticks or Markdown fences.
        """

    user_instructions = f"""
        TASK ({benchmark})
        ------------------
        {base_prompt}

        Return JSON:
        {response_schema}

        """.strip()


    # Return ONLY JSON with keys: plan, {expected_key}, explanation.
    # No markdown fences.
    # """.strip()

    response = gemini_client.models.generate_content(
        model=model,
        contents=types.Part.from_text(text=user_instructions),
        # config=types.GenerateContentConfig(temperature=0.3),
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

    raw_response = response.to_dict() if hasattr(response, "to_dict") else {"raw": str(response)}

    return InitialCodeResult(
        benchmark=benchmark,
        task_id=tid,
        model=model,
        plan=str(parsed.get("plan", "") or "").strip(),
        code=str(parsed.get(expected_key, "") or "").strip(),
        explanation=str(parsed.get("explanation", "") or "").strip(),
        raw_response=raw_response,
        raw_prompt=user_instructions,
    )


def generate_initial(task: TaskType, provider: str, model_name: str) -> InitialCodeResult:
    if provider == "openai":
        return generate_initial_code_with_openai(task, model=model_name)
    if provider == "gemini":
        return generate_initial_code_with_gemini(task, model=model_name)
    raise ValueError(f"Unknown provider: {provider}")


def _print_task_header(task_idx: int) -> None:
    print("\n" + "-" * 60)
    print(f"TASK {task_idx}")
    print("-" * 60)


def _print_iter_header(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)



# Self-debug stream (single patch model)
def self_debug_stream(
    *,
    task: TaskType,
    initial_code: str,
    first_exec: ExecutionResult,
    max_self_debug_iters: int,
    patch_model: str,
) -> Dict[str, Any]:
    if first_exec.passed or first_exec.num_tests == 0:
        return {
            "self_debug_used": False,
            "num_iterations": 1,
            "patch_models_used": [],
            "patch_explanations": [],
            "initial_code": initial_code,
            "final_code": initial_code,
            "passed": first_exec.passed,
            "num_tests": first_exec.num_tests,
            "num_passed": first_exec.num_passed,
            "error_type": first_exec.error_type,
            "error_message": first_exec.error_message,
            "traceback_str": first_exec.traceback_str,
            "stdout": first_exec.stdout,
            "stderr": first_exec.stderr,
            "initial_error_type": first_exec.error_type,
            "initial_error_message": first_exec.error_message,
            "iterations": [],
        }

    current = initial_code
    final_exec_result: ExecutionResult = first_exec

    patch_explanations: List[str] = []
    patch_models_used: List[str] = []
    iterations: List[Dict[str, Any]] = []

    for it in range(1, max_self_debug_iters + 1):
        _print_iter_header(f"Self-debug iteration {it}")

        feedback = build_feedback_package(task, current, final_exec_result)
        io_bundle = build_error_explanation_io(task, current, feedback)
        diag = diagnose_bug_with_openai(io_bundle)
        err_expl = build_error_explanation_text(io_bundle, diag)

        next_candidate, patch_info = produce_next_code_version(task, current, err_expl, model=patch_model)
        patch_text = getattr(patch_info, "rationale", None) or ""
        current = next_candidate

        print("Patch explanation:")
        print(patch_text.strip() if patch_text.strip() else "(none)")
        print("\nUpdated code:")
        print(current)

        patch_models_used.append(f"openai:{patch_model}")
        patch_explanations.append(patch_text)

        exec_result = execute_task(task, current)
        final_exec_result = exec_result

        iterations.append(
            {
                "iteration": it,
                "patch_model": patch_model,
                "patch_explanation": patch_text,
                "updated_code": current,
                "exec_result": {
                    "passed": exec_result.passed,
                    "num_tests": exec_result.num_tests,
                    "num_passed": exec_result.num_passed,
                    "error_type": exec_result.error_type,
                    "error_message": exec_result.error_message,
                    "traceback_str": exec_result.traceback_str,
                    "stdout": exec_result.stdout,
                    "stderr": exec_result.stderr,
                },
            }
        )

        if exec_result.passed or exec_result.num_tests == 0:
            print("\nTask solved")
            break

    return {
        "self_debug_used": True,
        "num_iterations": 1 + len(iterations),
        "patch_models_used": patch_models_used,
        "patch_explanations": patch_explanations,
        "initial_code": initial_code,
        "final_code": current,
        "passed": final_exec_result.passed,
        "num_tests": final_exec_result.num_tests,
        "num_passed": final_exec_result.num_passed,
        "error_type": final_exec_result.error_type,
        "error_message": final_exec_result.error_message,
        "traceback_str": final_exec_result.traceback_str,
        "stdout": final_exec_result.stdout,
        "stderr": final_exec_result.stderr,
        "initial_error_type": first_exec.error_type,
        "initial_error_message": first_exec.error_message,
        "iterations": iterations,
    }


# Sequential handoff stream (multiple patch agents)
def sequential_handoff_stream(
    *,
    task: TaskType,
    initial_code: str,
    first_exec: ExecutionResult,
    patch_agents: List[AgentSpec],
    max_iters: int,
) -> Dict[str, Any]:
    if first_exec.passed or first_exec.num_tests == 0:
        return {
            "handoff_used": False,
            "num_iterations": 1,
            "patch_models_used": [],
            "patch_explanations": [],
            "initial_code": initial_code,
            "final_code": initial_code,
            "passed": first_exec.passed,
            "num_tests": first_exec.num_tests,
            "num_passed": first_exec.num_passed,
            "error_type": first_exec.error_type,
            "error_message": first_exec.error_message,
            "iterations": [],
        }

    current = initial_code
    final_exec_result: ExecutionResult = first_exec

    patch_explanations: List[str] = []
    patch_models_used: List[str] = []
    iterations: List[Dict[str, Any]] = []

    for it in range(1, max_iters + 1):
        fixer = pick_fixer(patch_agents, it)
        if fixer.provider != "openai":
            raise ValueError("Sequential handoff currently supports OpenAI patch agents only.")

        _print_iter_header(f"Sequential handoff iteration {it}  (fixer: {fixer.model})")

        feedback = build_feedback_package(task, current, final_exec_result)
        io_bundle = build_error_explanation_io(task, current, feedback)
        diag = diagnose_bug_with_openai(io_bundle)
        err_expl = build_error_explanation_text(io_bundle, diag)

        next_candidate, patch_info = produce_next_code_version(task, current, err_expl, model=fixer.model)
        patch_text = getattr(patch_info, "rationale", None) or ""
        current = next_candidate

        print("Patch explanation:")
        print(patch_text.strip() if patch_text.strip() else "(none)")
        print("\nUpdated code:")
        print(current)

        patch_models_used.append(f"openai:{fixer.model}")
        patch_explanations.append(patch_text)

        exec_result = execute_task(task, current)
        final_exec_result = exec_result

        iterations.append(
            {
                "iteration": it,
                "fixer": {"provider": fixer.provider, "model": fixer.model},
                "patch_explanation": patch_text,
                "updated_code": current,
                "exec_result": {
                    "passed": exec_result.passed,
                    "num_tests": exec_result.num_tests,
                    "num_passed": exec_result.num_passed,
                    "error_type": exec_result.error_type,
                    "error_message": exec_result.error_message,
                    "traceback_str": exec_result.traceback_str,
                    "stdout": exec_result.stdout,
                    "stderr": exec_result.stderr,
                },
            }
        )

        if exec_result.passed or exec_result.num_tests == 0:
            print("\nTask solved (handoff)")
            break

    return {
        "handoff_used": True,
        "num_iterations": 1 + len(iterations),
        "patch_models_used": patch_models_used,
        "patch_explanations": patch_explanations,
        "initial_code": initial_code,
        "final_code": current,
        "passed": final_exec_result.passed,
        "num_tests": final_exec_result.num_tests,
        "num_passed": final_exec_result.num_passed,
        "error_type": final_exec_result.error_type,
        "error_message": final_exec_result.error_message,
        "iterations": iterations,
    }



# Plotting
def shorten(s: str, max_len: int = 26) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


# def add_value_labels(ax, bars):
#     for b in bars:
#         h = float(b.get_height() or 0.0)
#         ax.annotate(
#             f"{h:.1f}%",
#             (b.get_x() + b.get_width() / 2, h),
#             textcoords="offset points",
#             xytext=(0, 3),
#             ha="center",
#             va="bottom",
#             fontsize=9,
#         )

def _benchmark_task_counts_from_results(all_results: List[tuple]) -> Dict[str, int]:
    """
    Returns {benchmark: N} where N is the maximum num_tasks observed for that benchmark
    (across providers/models/modes). This reflects actual loaded tasks, not just max_tasks.
    """
    counts: Dict[str, int] = {}
    for _, summary in all_results:
        bench = summary.get("benchmark")
        n = int(summary.get("num_tasks") or 0)
        if bench:
            counts[bench] = max(counts.get(bench, 0), n)
    return counts


def plot_clean_grouped_bars(all_results: List[tuple], k_values: List[int], out_dir: str) -> List[str]:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    bench_counts = _benchmark_task_counts_from_results(all_results)

    saved_paths: List[str] = []
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
            # add_value_labels(ax, bars)

        n_tasks = bench_counts.get(benchmark, 0)    
        ax.set_title(f"{benchmark} (N={n_tasks}): Baseline vs Single Self-Debug vs Sequential Handoff")
        ax.set_ylabel("Pass Rate (%)")
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

        fig.tight_layout()
        out_file = outp / f"pass_rates_{benchmark}_clean.png".replace("/", "_")
        plt.savefig(out_file, dpi=220, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(str(out_file))

    return saved_paths


def plot_improvement_over_baseline(all_results: List[tuple], k_values: List[int], out_dir: str) -> List[str]:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    bench_counts = _benchmark_task_counts_from_results(all_results)

    saved_paths: List[str] = []
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
        n_tasks = bench_counts.get(bench, 0)
        ax.set_title(f"{bench} (N={n_tasks}): Improvement over Baseline vs # Patch Agents (Sequential Handoff)")
        ax.set_xlabel("Number of patch agents (K)")
        ax.set_ylabel("Pass Rate (percentage points)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

        fig.tight_layout()
        out_file = outp / f"improvement_vs_k_{bench}.png".replace("/", "_")
        plt.savefig(out_file, dpi=220, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(str(out_file))

    return saved_paths


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




if __name__ == "__main__":
    ts, run_dir, plots_dir = make_run_dir("results")

    print("\n" + "=" * 80)
    print("STARTING EXPERIMENT")
    print("=" * 80)

    configs = [
        # ("gemini", "gemini-3-flash-preview"),
        ("openai", "gpt-4o"),
    ]
    # benchmarks = ["HumanEval", "MBPP", "APPS", "SWE-bench_LITE"]
    benchmarks = ["SWE-bench_LITE"]
    max_tasks = 200
    max_self_debug_iters = 5

    single_patch_model = "gpt-4o"

    patch_pool: List[AgentSpec] = [
        AgentSpec("openai", "gpt-4.1-mini"),
        AgentSpec("openai", "gpt-4.1"),
        AgentSpec("openai", "gpt-5-mini"),
        AgentSpec("openai", "gpt-5"),
    ]
    k_values = [2,3]

    all_results: List[Tuple[str, Dict[str, Any]]] = []
    all_details: Dict[str, Any] = {"baseline": {}, "self_debug_single": {}, "sequential_handoff": {}}
    compact_summary_rows: List[Dict[str, Any]] = []

    for benchmark in benchmarks:
        for provider, model_name in configs:
            print("\n" + "=" * 80)
            print(f"{benchmark} :: {provider}:{model_name}")
            print("=" * 80)

            tasks = load_tasks_for_benchmark(benchmark, max_tasks)

            baseline_details: List[Dict[str, Any]] = []
            self_debug_details: List[Dict[str, Any]] = []
            handoff_details_by_k: Dict[int, List[Dict[str, Any]]] = {k: [] for k in k_values}

            for idx, task in enumerate(tasks, start=1):
                _print_task_header(idx)

                # 1) initial generation
                init = generate_initial(task, provider, model_name)
                candidate = init.code
                # print(init)
                # print("==" * 30)
                # print(init.code)

                # break

                print("Initial raw prompt:\n")
                print(init.raw_prompt)

                print("\n\nInitial plan:")
                print(init.plan if init.plan.strip() else "(none)")

                print("\n\nInitial code: ------------")
                print(candidate if str(candidate).strip() else "(empty)")

                # 2) baseline
                base_exec = execute_task(task, candidate)

                print("\n\nBaseline result:")
                print(f"Passed: {base_exec.passed}")
                print(f"Initial error type: {base_exec.error_type}")
                print(f"Initial error message: {base_exec.error_message}")

                baseline_details.append(
                    {
                        "timestamp": ts,
                        "benchmark": base_exec.benchmark,
                        "task_id": base_exec.task_id,
                        "provider": provider,
                        "model": model_name,
                        "mode_tag": "baseline",
                        "passed": base_exec.passed,
                        "num_tests": base_exec.num_tests,
                        "num_passed": base_exec.num_passed,
                        "artifact": base_exec.patch,
                        "error_type": base_exec.error_type,
                        "error_message": base_exec.error_message,
                        "traceback_str": base_exec.traceback_str,
                        "stdout": base_exec.stdout,
                        "stderr": base_exec.stderr,
                        "initial_raw_prompt": init.raw_prompt,
                        "initial_plan": init.plan,
                        "initial_code": candidate,
                        "final_code": candidate,
                        "prompt": task.build_prompt() if hasattr(task, "build_prompt") else "",
                    }
                )

                # If baseline already passed, skip add self-debug and handoff
                if base_exec.passed or base_exec.num_tests == 0:
                    sd_noop = {
                        "timestamp": ts,
                        "benchmark": base_exec.benchmark,
                        "task_id": base_exec.task_id,
                        "provider": provider,
                        "model": model_name,
                        "mode_tag": "self_debug_single",
                        "passed": base_exec.passed,
                        "num_tests": base_exec.num_tests,
                        "num_passed": base_exec.num_passed,
                        "error_type": base_exec.error_type,
                        "error_message": base_exec.error_message,
                        "initial_error_type": base_exec.error_type,
                        "initial_error_message": base_exec.error_message,
                        "initial_raw_prompt": init.raw_prompt,
                        "initial_plan": init.plan,
                        "initial_code": candidate,
                        "final_code": candidate,
                        "patch_models_used": [],
                        "patch_explanations": [],
                        "iterations": [],
                        "traceback_str": base_exec.traceback_str,
                        "stdout": base_exec.stdout,
                        "stderr": base_exec.stderr,
                        "prompt": task.build_prompt() if hasattr(task, "build_prompt") else "",
                    }
                    self_debug_details.append(sd_noop)

                    for k in k_values:
                        handoff_details_by_k[k].append(
                            {
                                "timestamp": ts,
                                "benchmark": base_exec.benchmark,
                                "task_id": base_exec.task_id,
                                "provider": provider,
                                "model": model_name,
                                "mode_tag": f"handoff_{k}agents",
                                "passed": base_exec.passed,
                                "num_tests": base_exec.num_tests,
                                "num_passed": base_exec.num_passed,
                                "error_type": base_exec.error_type,
                                "error_message": base_exec.error_message,
                                "initial_raw_prompt": init.raw_prompt,
                                "initial_plan": init.plan,
                                "initial_code": candidate,
                                "final_code": candidate,
                                "patch_models_used": [],
                                "patch_explanations": [],
                                "iterations": [],
                                "prompt": task.build_prompt() if hasattr(task, "build_prompt") else "",
                            }
                        )
                    continue

                # 3) self-debug only if baseline failed
                sd = self_debug_stream(
                    task=task,
                    initial_code=candidate,
                    first_exec=base_exec,
                    max_self_debug_iters=max_self_debug_iters,
                    patch_model=single_patch_model,
                )

                print("\n\nFinal code:")
                print(sd["final_code"])

                if sd.get("patch_explanations"):
                    non_empty = [p for p in sd["patch_explanations"] if str(p).strip()]
                    if non_empty:
                        print("\n\nPatch explanations:")
                        for p in non_empty:
                            print(f"- {p.strip()}")

                self_debug_details.append(
                    {
                        "timestamp": ts,
                        "benchmark": base_exec.benchmark,
                        "task_id": base_exec.task_id,
                        "provider": provider,
                        "model": model_name,
                        "mode_tag": "self_debug_single",
                        "passed": sd["passed"],
                        "num_tests": sd["num_tests"],
                        "num_passed": sd["num_passed"],
                        "error_type": sd.get("error_type"),
                        "error_message": sd.get("error_message"),
                        "initial_error_type": sd.get("initial_error_type"),
                        "initial_error_message": sd.get("initial_error_message"),
                        "initial_raw_prompt": init.raw_prompt,
                        "initial_plan": init.plan,
                        "initial_code": candidate,
                        "final_code": sd["final_code"],
                        "patch_models_used": sd.get("patch_models_used", []),
                        "patch_explanations": sd.get("patch_explanations", []),
                        "iterations": sd.get("iterations", []),
                        "traceback_str": sd.get("traceback_str"),
                        "stdout": sd.get("stdout"),
                        "stderr": sd.get("stderr"),
                        "prompt": task.build_prompt() if hasattr(task, "build_prompt") else "",
                    }
                )

                # If self-debug solved, do nnot run handoff; log as skipped for each k
                if sd.get("passed") or sd.get("num_tests", 0) == 0:
                    for k in k_values:
                        handoff_details_by_k[k].append(
                            {
                                "timestamp": ts,
                                "benchmark": base_exec.benchmark,
                                "task_id": base_exec.task_id,
                                "provider": provider,
                                "model": model_name,
                                "mode_tag": f"handoff_{k}agents",
                                "passed": sd["passed"],
                                "num_tests": sd["num_tests"],
                                "num_passed": sd["num_passed"],
                                "error_type": sd.get("error_type"),
                                "error_message": sd.get("error_message"),
                                "initial_raw_prompt": init.raw_prompt,
                                "initial_plan": init.plan,
                                "initial_code": candidate,
                                "final_code": sd["final_code"],
                                "patch_models_used": [],
                                "patch_explanations": [],
                                "iterations": [],
                                "prompt": task.build_prompt() if hasattr(task, "build_prompt") else "",
                                "skipped_reason": "self_debug_solved",
                            }
                        )
                    continue

                # 4) Run sequential handoff only if self-debug failed
                for k in k_values:
                    agents_k = patch_pool[:k]
                    handoff = sequential_handoff_stream(
                        task=task,
                        initial_code=candidate,
                        first_exec=base_exec,
                        patch_agents=agents_k,
                        max_iters=max_self_debug_iters,
                    )

                    handoff_details_by_k[k].append(
                        {
                            "timestamp": ts,
                            "benchmark": base_exec.benchmark,
                            "task_id": base_exec.task_id,
                            "provider": provider,
                            "model": model_name,
                            "mode_tag": f"handoff_{k}agents",
                            "passed": handoff["passed"],
                            "num_tests": handoff["num_tests"],
                            "num_passed": handoff["num_passed"],
                            "error_type": handoff.get("error_type"),
                            "error_message": handoff.get("error_message"),
                            "initial_raw_prompt": init.raw_prompt,
                            "initial_plan": init.plan,
                            "initial_code": candidate,
                            "final_code": handoff["final_code"],
                            "patch_models_used": handoff.get("patch_models_used", []),
                            "patch_explanations": handoff.get("patch_explanations", []),
                            "iterations": handoff.get("iterations", []),
                            "prompt": task.build_prompt() if hasattr(task, "build_prompt") else "",
                        }
                    )

            # summaries (benchmark, provider, model)
            baseline_summary = summarize_results(baseline_details, benchmark, provider, model_name, "baseline")
            self_debug_summary = summarize_results(self_debug_details, benchmark, provider, model_name, "self_debug_single")

            all_results.append(("baseline", baseline_summary))
            all_results.append(("self_debug_single", self_debug_summary))

            for k in k_values:
                tag = f"handoff_{k}agents"
                summary_k = summarize_results(handoff_details_by_k[k], benchmark, provider, model_name, tag)
                all_results.append((tag, summary_k))

            run_key = f"{benchmark}::{provider}::{model_name}"
            all_details["baseline"][run_key] = baseline_details
            all_details["self_debug_single"][run_key] = self_debug_details
            all_details["sequential_handoff"][run_key] = {str(k): handoff_details_by_k[k] for k in k_values}

            compact_row = {
                "benchmark": benchmark,
                "provider": provider,
                "model": model_name,
                "baseline": baseline_summary,
                "self_debug": self_debug_summary,
                "handoff_by_k": {
                    k: next(
                        s
                        for t, s in all_results
                        if t == f"handoff_{k}agents"
                        and s["benchmark"] == benchmark
                        and s["provider"] == provider
                        and s["model"] == model_name
                    )
                    for k in k_values
                },
            }
            compact_summary_rows.append(compact_row)

    # plot graphs
    pass_rate_plot_paths = plot_clean_grouped_bars(all_results, k_values, plots_dir)
    improvement_plot_paths = plot_improvement_over_baseline(all_results, k_values, plots_dir)

    experiment_config = {
        "benchmarks": benchmarks,
        "configs": configs,
        "max_tasks": max_tasks,
        "max_self_debug_iters": max_self_debug_iters,
        "single_patch_model": single_patch_model,
        "patch_pool": [{"provider": a.provider, "model": a.model} for a in patch_pool],
        "k_values": k_values,
        "openai_default_model": OPENAI_MODEL,
        "gemini_default_model": GOOGLE_MODEL,
    }

    summary_report_text = build_summary_report_text(compact_summary_rows)

    artifacts = {
        "run_dir": run_dir,
        "plots_dir": plots_dir,
        "plots": {
            "pass_rates": pass_rate_plot_paths,
            "improvement_vs_k": improvement_plot_paths,
        },
        "summary_report_text": summary_report_text,
    }

    saved_path = save_experiment_results(
        run_dir=run_dir,
        timestamp=ts,
        summaries=all_results,
        details=all_details,
        config=experiment_config,
        artifacts=artifacts,
    )

    print("\n")
    for row in compact_summary_rows:
        bench = row["benchmark"]
        provider = row["provider"]
        model = row["model"]

        bsum = row["baseline"]
        ssum = row["self_debug"]

        print(f"{bench} on {provider}:{model}")
        print(f"Baseline: {bsum['num_passed']}/{bsum['num_tasks']} ({bsum['pass_rate']*100:.2f}%)")
        print(f"Self-debug (only on failures(baseline)): {ssum['num_passed']}/{ssum['num_tasks']} ({ssum['pass_rate']*100:.2f}%)")

        for k, hsum in row["handoff_by_k"].items():
            print(f"Handoff ({k}): {hsum['num_passed']}/{hsum['num_tasks']} ({hsum['pass_rate']*100:.2f}%)")

        print("")

    print(f"[LOG] Results saved to {saved_path}")
    print("\nEXPERIMENT COMPLETE")
