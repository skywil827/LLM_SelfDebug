import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, List, Literal, Tuple
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
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
    diagnose_bug_with_gemini,
    diagnose_bug_with_claude,
    diagnose_bug_with_ollama,
    build_error_explanation_text,
)
from patch_generation import produce_next_code_version
from ollama import Client


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def make_run_dir(out_root: str = "results") -> Tuple[str, str, str]:
    ts = _now_ts()
    run_dir = Path(out_root) / f"run_{ts}"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return ts, str(run_dir), str(plots_dir)


def build_summary_report_text(compact_summary_rows: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for row in compact_summary_rows:
        bench = row["benchmark"]
        provider = row["provider"]
        model = row["model"]

        bsum = row["baseline"]
        ssum = row["self_debug"]

        lines.append(f"{bench} on {provider}:{model}")
        lines.append(f"Baseline: {bsum['num_passed']}/{bsum['num_tasks']} ({bsum['pass_rate']*100:.2f}%)")
        lines.append(
            f"Self-debug (only on failures): {ssum['num_passed']}/{ssum['num_tasks']} ({ssum['pass_rate']*100:.2f}%)"
        )

        for k, hsum in row["handoff_by_k"].items():
            lines.append(f"Handoff ({k}): {hsum['num_passed']}/{hsum['num_tasks']} ({hsum['pass_rate']*100:.2f}%)")

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
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
ollama_api_key = os.getenv("OLLAMA_API_KEY")
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

openai_client = OpenAI()
gemini_client = genai.Client(api_key=google_api_key)
claude_client = anthropic.Anthropic(api_key=anthropic_api_key)

ollama_headers = {}
if ollama_api_key:
    ollama_headers["Authorization"] = f"Bearer {ollama_api_key}"

ollama_client = Client(
    host=ollama_host,
    headers=ollama_headers if ollama_headers else None,
)

OPENAI_MODEL = "gpt-5.4"
GOOGLE_MODEL = "gemini-3.1-pro-preview"
CLAUDE_MODEL = "claude-sonnet-4-6"
OLLAMA_MODEL = "qwen2.5-coder:3b"

TaskType = Union[HumanEvalTask, MBPPTask, APPSTask, SWELITETask]
Provider = Literal["openai", "gemini", "anthropic", "ollama"]


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
    constraints = getattr(task, "constraints", {})
    benchmark = constraints.get("benchmark", "UNKNOWN")

    if hasattr(task, "task_id") and getattr(task, "task_id") is not None:
        tid = str(getattr(task, "task_id"))
        if benchmark == "MBPP" and not tid.startswith("MBPP/"):
            tid = f"MBPP/{tid}"
        return benchmark, tid

    elif hasattr(task, "problem_id") and getattr(task, "problem_id") is not None:
        return benchmark, f"APPS/{getattr(task, 'problem_id')}"

    elif hasattr(task, "instance_id") and getattr(task, "instance_id") is not None:
        return benchmark, f"SWELITE/{getattr(task, 'instance_id')}"
    else:
        return benchmark, "UNKNOWN"


def debug_print_raw_model_content(provider: str, content: str) -> None:
    print("\n[Raw Model Content]")
    print(f"Provider: {provider}")
    print(repr(content))
    print()


def _make_empty_generation_result(task: TaskType, provider: str, model_name: str) -> ExecutionResult:
    benchmark, tid = _get_task_identity(task)
    return ExecutionResult(
        benchmark=benchmark,
        task_id=tid,
        patch="",
        passed=False,
        num_tests=0,
        num_passed=0,
        error_type="EmptyGeneration",
        error_message=f"Model returned empty code for {benchmark}:{tid} using {provider}:{model_name}",
        traceback_str="",
        stdout="",
        stderr="",
    )


def safe_execute_generated_code(
    task: TaskType,
    generated_code: str,
    provider: str,
    model_name: str,
) -> ExecutionResult:
    if not (generated_code or "").strip():
        print("\n[Empty Generation Detected]")
        print(f"Provider: {provider}")
        print(f"Model: {model_name}")
        print("Generated code is empty.\n")
        return _make_empty_generation_result(task, provider, model_name)

    return execute_task(task, generated_code)


# Initial generation
def generate_initial_code_with_openai(task: TaskType, model: str = OPENAI_MODEL) -> InitialCodeResult:
    benchmark, tid = _get_task_identity(task)

    if isinstance(task, SWELITETask):
        base_prompt = build_swe_prompt(task)
        artifact_key = "patch"
        response_schema = """{ "plan": "...", "patch": "...", "explanation": "..." }"""
        artifact_instruction = (
            'The "patch" field must contain the full corrected unified diff patch only. '
            "It must be complete and directly usable."
        )
    else:
        base_prompt = task.build_prompt()
        artifact_key = "code"
        response_schema = """{ "plan": "...", "code": "...", "explanation": "..." }"""
        artifact_instruction = (
            'The "code" field must contain complete, self-contained Python code. '
            "Include all required imports needed for execution"
            "Do not omit imports from the problem specification if the generated code depends on them."
        )

    system_msg = "You are a highly reliable coding assistant. Follow the schema and return JSON only."
    user_instructions = f"""
        TASK ({benchmark})
        ------------------
        {base_prompt}
        
        Return ONLY valid JSON
        {artifact_instruction}

        Return JSON:
        {response_schema}
    """.strip()

    response = openai_client.chat.completions.create(
        model=model,
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

    if isinstance(task, SWELITETask):
        base_prompt = build_swe_prompt(task)
        artifact_key = "patch"
        response_schema = """{ "plan": "...", "patch": "...", "explanation": "..." }"""
        artifact_instruction = (
            'The "patch" field must contain the full corrected unified diff patch only. '
            "It must be complete and directly usable."
        )
    else:
        base_prompt = task.build_prompt()
        artifact_key = "code"
        response_schema = """{ "plan": "...", "code": "...", "explanation": "..." }"""
        artifact_instruction = (
            'The "code" field must contain complete, self-contained Python code. '
            "Include all required imports needed for execution"
            "Do not omit imports from the problem specification if the generated code depends on them."
        )

    system_msg = "You are a highly reliable coding assistant. Follow the schema and return JSON only."
    user_instructions = f"""
        TASK ({benchmark})
        ------------------
        {base_prompt}
        
        Return ONLY valid JSON
        {artifact_instruction}

        Return JSON:
        {response_schema}
    """.strip()


    response = gemini_client.models.generate_content(
        model=model,
        contents=types.Part.from_text(text=user_instructions),
    )

    content = (getattr(response, "text", "") or str(response)).strip()
    if content.startswith("```"):
        lines = content.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"plan": "", artifact_key: "", "explanation": content}

    raw_response = response.to_dict() if hasattr(response, "to_dict") else {"raw": str(response)}

    return InitialCodeResult(
        benchmark=benchmark,
        task_id=tid,
        model=model,
        plan=str(parsed.get("plan", "") or "").strip(),
        code=str(parsed.get(artifact_key, "") or "").strip(),
        explanation=str(parsed.get("explanation", "") or "").strip(),
        raw_response=raw_response,
        raw_prompt=user_instructions,
    )


def generate_initial_code_with_claude(task: TaskType, model: str = CLAUDE_MODEL) -> InitialCodeResult:
    benchmark, tid = _get_task_identity(task)

    if isinstance(task, SWELITETask):
        base_prompt = build_swe_prompt(task)
        artifact_key = "patch"
        response_schema = """{ "plan": "...", "patch": "...", "explanation": "..." }"""
        artifact_instruction = (
            'The "patch" field must contain the full corrected unified diff patch only. '
            "It must be complete and directly usable."
        )
    else:
        base_prompt = task.build_prompt()
        artifact_key = "code"
        response_schema = """{ "plan": "...", "code": "...", "explanation": "..." }"""
        artifact_instruction = (
            'The "code" field must contain complete, self-contained Python code. '
            "Include all required imports needed for execution"
            "Do not omit imports from the problem specification if the generated code depends on them."
        )

    system_msg = "You are a highly reliable coding assistant. Follow the schema and return JSON only."
    user_instructions = f"""
        TASK ({benchmark})
        ------------------
        {base_prompt}
        
        Return ONLY valid JSON
        {artifact_instruction}

        Return JSON:
        {response_schema}
    """.strip()

    json_schema = {
        "type": "object",
        "properties": {
            "plan": {"type": "string"},
            "code": {"type": "string"},
            "patch": {"type": "string"},
            "explanation": {"type": "string"},
        },
        "required": ["plan", artifact_key, "explanation"],
        "additionalProperties": False,
    }

    response = claude_client.messages.create(
        model=model,
        max_tokens=6000,
        system=system_msg,
        messages=[{"role": "user", "content": user_instructions}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": json_schema,
            }
        },
    )

    content = (response.content[0].text or "").strip()

    if content.startswith("```"):
        lines = content.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        print("\n[CLAUDE JSON ERROR]")
        print(f"Failed to parse Claude output: {e}")
        print("Raw content:")
        print(content)

        parsed = {
            "plan": "",
            artifact_key: "",
            "explanation": content,
        }

    return InitialCodeResult(
        benchmark=benchmark,
        task_id=tid,
        model=model,
        plan=str(parsed.get("plan", "")).strip(),
        code=str(parsed.get(artifact_key, "")).strip(),
        explanation=str(parsed.get("explanation", "")).strip(),
        raw_response=response.model_dump() if hasattr(response, "model_dump") else {"raw": str(response)},
        raw_prompt=user_instructions,
    )


def generate_initial_code_with_ollama(task: TaskType, model: str = OLLAMA_MODEL) -> InitialCodeResult:
    benchmark, tid = _get_task_identity(task)

    if isinstance(task, SWELITETask):
        base_prompt = build_swe_prompt(task)
        artifact_key = "patch"
        response_schema = """{ "plan": "...", "patch": "...", "explanation": "..." }"""
        artifact_instruction = (
            'The "patch" field must contain the full corrected unified diff patch only. '
            "It must be complete and directly usable."
        )
    else:
        base_prompt = task.build_prompt()
        artifact_key = "code"
        response_schema = """{ "plan": "...", "code": "...", "explanation": "..." }"""
        artifact_instruction = (
            'The "code" field must contain complete, self-contained Python code. '
            "Include all required imports needed for execution"
            "Do not omit imports from the problem specification if the generated code depends on them."
        )

    system_msg = "You are a highly reliable coding assistant. Follow the schema and return JSON only."
    user_instructions = f"""
        TASK ({benchmark})
        ------------------
        {base_prompt}
        
        Return ONLY valid JSON
        {artifact_instruction}

        Return JSON:
        {response_schema}
    """.strip()


    schema = {
        "type": "object",
        "properties": {
            "plan": {"type": "string"},
            "code": {"type": "string"},
            "patch": {"type": "string"},
            "explanation": {"type": "string"},
        },
        "required": ["plan", artifact_key, "explanation"],
        "additionalProperties": False,
    }

    response = ollama_client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_instructions},
        ],
        stream=False,
        format=schema,
        options={"temperature": 0},
    )

    content = (response.get("message", {}) or {}).get("content", "") or ""
    content = content.strip()

    debug_print_raw_model_content("ollama", content)

    if content.startswith("```"):
        lines = content.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

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
        raw_response=response,
        raw_prompt=user_instructions,
    )


def generate_initial(task: TaskType, provider: str, model_name: str) -> InitialCodeResult:
    if provider == "openai":
        return generate_initial_code_with_openai(task, model=model_name)
    elif provider == "gemini":
        return generate_initial_code_with_gemini(task, model=model_name)
    elif provider == "anthropic":
        return generate_initial_code_with_claude(task, model=model_name)
    elif provider == "ollama":
        return generate_initial_code_with_ollama(task, model=model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _print_task_header(task_idx: int) -> None:
    print("\n" + "-" * 60)
    print(f"TASK {task_idx}")
    print("-" * 60)


def _print_iter_header(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


# Self-debug stream
def self_debug_stream(
    *,
    task: TaskType,
    initial_code: str,
    first_exec: ExecutionResult,
    max_self_debug_iters: int,
    provider: Provider,
    patch_model: str,
) -> Dict[str, Any]:
    if first_exec.passed or first_exec.num_tests == 0 or max_self_debug_iters <= 0:
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

        if provider == "openai":
            diag = diagnose_bug_with_openai(io_bundle)
        elif provider == "gemini":
            diag = diagnose_bug_with_gemini(io_bundle)
        elif provider == "anthropic":
            diag = diagnose_bug_with_claude(io_bundle)
        elif provider == "ollama":
            diag = diagnose_bug_with_ollama(io_bundle)
        else:
            raise ValueError(f"Unknown provider in self_debug_stream: {provider}")

        err_expl = build_error_explanation_text(io_bundle, diag)

        next_candidate, patch_info = produce_next_code_version(
            task, current, err_expl, provider=provider, model=patch_model
        )

        patch_text = getattr(patch_info, "rationale", None) or ""
        current = next_candidate

        print("Patch explanation:")
        print(patch_text.strip() if patch_text.strip() else "(none)")
        print("\nUpdated code:")
        print(current)

        patch_models_used.append(f"{provider}:{patch_model}")
        patch_explanations.append(patch_text)

        exec_result = safe_execute_generated_code(
            task=task,
            generated_code=current,
            provider=provider,
            model_name=patch_model,
        )
        final_exec_result = exec_result

        iterations.append(
            {
                "iteration": it,
                "patch_model": f"{provider}:{patch_model}",
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


# Sequential handoff
def sequential_handoff_stream(
    *,
    task: TaskType,
    initial_code: str,
    first_exec: ExecutionResult,
    patch_agents: List[AgentSpec],
    max_iters: int,
) -> Dict[str, Any]:
    if first_exec.passed or first_exec.num_tests == 0 or max_iters <= 0:
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

        _print_iter_header(f"Sequential handoff iteration {it}  (fixer: {fixer.provider}:{fixer.model})")
        feedback = build_feedback_package(task, current, final_exec_result)
        io_bundle = build_error_explanation_io(task, current, feedback)

        if fixer.provider == "openai":
            diag = diagnose_bug_with_openai(io_bundle)
        elif fixer.provider == "gemini":
            diag = diagnose_bug_with_gemini(io_bundle)
        elif fixer.provider == "anthropic":
            diag = diagnose_bug_with_claude(io_bundle)
        elif fixer.provider == "ollama":
            diag = diagnose_bug_with_ollama(io_bundle)
        else:
            raise ValueError("Sequential handoff failed to select provider.")

        err_expl = build_error_explanation_text(io_bundle, diag)

        next_candidate, patch_info = produce_next_code_version(
            task, current, err_expl, provider=fixer.provider, model=fixer.model
        )

        patch_text = getattr(patch_info, "rationale", None) or ""
        current = next_candidate

        print("Patch explanation:")
        print(patch_text.strip() if patch_text.strip() else "(none)")
        print("\nUpdated code:")
        print(current)

        patch_models_used.append(f"{fixer.provider}:{fixer.model}")
        patch_explanations.append(patch_text)

        exec_result = safe_execute_generated_code(
            task=task,
            generated_code=current,
            provider=fixer.provider,
            model_name=fixer.model,
        )
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


def _benchmark_task_counts_from_results(all_results: List[tuple]) -> Dict[str, int]:
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
        models_by_mode = defaultdict(list)

        for mode_tag, summary in entries:
            key = (summary["provider"], summary["model"])
            grouped[key][mode_tag] = summary["pass_rate"] * 100.0
            models_by_mode[mode_tag].append(f"{summary['provider']}:{summary['model']}")

        keys = list(grouped.keys())
        labels = [shorten(f"{p}:{m}", 28) for (p, m) in keys]
        series_vals = {s: [grouped[k].get(s, 0.0) for k in keys] for s in series_order}

        x = list(range(len(keys)))
        width = 0.15

        mid = (len(series_order) - 1) / 2.0
        offsets = [(i - mid) * width for i in range(len(series_order))]

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.6)

        for i, s in enumerate(series_order):
            ax.bar([xi + offsets[i] for xi in x], series_vals[s], width, label=s)

        n_tasks = bench_counts.get(benchmark, 0)
        ax.set_title(f"{benchmark} (N={n_tasks}): Baseline vs Single Self-Debug vs Sequential Handoff")
        ax.set_ylabel("Pass Rate (%)")
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)

        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

        left_text_lines = []

        left_text_lines.append("Baseline:")
        left_text_lines.extend(sorted(set(models_by_mode.get("baseline", []))) or ["  - None"])
        left_text_lines.append("")

        left_text_lines.append("Self-Debug:")
        left_text_lines.extend(sorted(set(models_by_mode.get("self_debug_single", []))) or ["  - None"])
        left_text_lines.append("")

        for k in k_values:
            tag = f"handoff_{k}agents"
            left_text_lines.append(f"Handoff ({k} agents):")
            left_text_lines.extend(sorted(set(models_by_mode.get(tag, []))) or ["  - None"])
            left_text_lines.append("")

        left_text = "\n".join(left_text_lines)

        fig.subplots_adjust(left=0.08, right=0.72)

        fig.text(
            0.75, 0.5, left_text,
            ha="left", va="center", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
        )

        out_file = outp / f"pass_rates_{benchmark}_clean.png".replace("/", "_")
        plt.savefig(out_file, dpi=220, bbox_inches="tight")
        plt.close(fig)

        saved_paths.append(str(out_file))

    return saved_paths


# Loading tasks + summaries
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


def handoff_model_label(provider: Provider, agents_k: List[AgentSpec]) -> str:
    models = ",".join(a.model for a in agents_k)
    return f"handoff[{provider}]:{models}"


def get_patch_agents_for_provider(
    provider: Provider,
    k: int,
    patch_pool: List[AgentSpec],
    gemini_patch_pool: List[AgentSpec],
    claude_patch_pool: List[AgentSpec],
    ollama_patch_pool: List[AgentSpec],
) -> List[AgentSpec]:
    if provider == "openai":
        return patch_pool[:k]
    if provider == "gemini":
        return gemini_patch_pool[:k]
    if provider == "anthropic":
        return claude_patch_pool[:k]
    if provider == "ollama":
        return ollama_patch_pool[:k]
    raise ValueError(f"Unknown provider for handoff: {provider}")


if __name__ == "__main__":
    # print(generate_initial_code_with_claude(load_humaneval_task('HumanEval/13')))
    # print(generate_initial_code_with_gemini(load_humaneval_task('HumanEval/13')))
    ts, run_dir, plots_dir = make_run_dir("results")

    print("\n" + "=" * 80)
    print("STARTING EXPERIMENT")
    print("=" * 80)

    configs: List[Tuple[Provider, str]] = [
        # ("gemini", "gemini-3.1-pro-preview"),
        # ("openai", "gpt-5.4"),
        # ("anthropic", "claude-opus-4-6"),
        ("ollama", "qwen2.5-coder:3b"),
    ]


    # benchmarks = ["HumanEval", "MBPP", "APPS", "SWE-bench_LITE"]
    benchmarks = ["SWE-bench_LITE"]

    max_tasks = 2
    max_self_debug_iters = 1

    patch_pool: List[AgentSpec] = [
        # AgentSpec("openai", "gpt-4.1-mini"),
        # AgentSpec("openai", "gpt-4.1"),
        # AgentSpec("openai", "gpt-5-mini"),
        AgentSpec("openai", "gpt-5.4"),
        AgentSpec("openai", "gpt-5.4-mini"),
        AgentSpec("openai", "gpt-5.4-nano"),
        AgentSpec("openai", "gpt-5"),
        AgentSpec("openai", "gpt-5-mini"),
    ]

    gemini_patch_pool: List[AgentSpec] = [
        AgentSpec("gemini", "gemini-3.1-pro-preview"),
        AgentSpec("gemini", "gemini-3-flash-preview"),
        AgentSpec("gemini", "gemini-3-pro-preview"),
        AgentSpec("gemini", "gemini-3.1-flash-lite-preview"),
        AgentSpec("gemini", "gemini-2.5-flash"),
    ]

    claude_patch_pool: List[AgentSpec] = [
        AgentSpec("anthropic", "claude-sonnet-4-6"),
        AgentSpec("anthropic", "claude-haiku-4-5"),
    ]

    ollama_patch_pool: List[AgentSpec] = [
        AgentSpec("ollama", "qwen2.5-coder:3b"),
        AgentSpec("ollama", "qwen2.5-coder:1.5b"),
        AgentSpec("ollama", "deepseek-coder:1.3b"),
    ]

    # ollama_patch_pool: List[AgentSpec] = [
    #     AgentSpec("ollama", "qwen2.5-coder:7b"),
    #     AgentSpec("ollama", "deepseek-coder:6.7b"),
    #     AgentSpec("ollama", "llama3:8b"),
    # ]


    k_values = [2]

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

                init = generate_initial(task, provider, model_name)
                candidate = init.code

                print("Initial raw prompt:\n")
                print(init.raw_prompt)

                print("\n\nInitial plan:")
                print(init.plan if init.plan.strip() else "(none)")

                print("\n\nInitial code: ------------")
                print(candidate if str(candidate).strip() else "(empty)")

                base_exec = safe_execute_generated_code(
                    task=task,
                    generated_code=candidate,
                    provider=provider,
                    model_name=model_name,
                )

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
                        agents_k = get_patch_agents_for_provider(
                            provider, k, patch_pool, gemini_patch_pool, claude_patch_pool, ollama_patch_pool
                        )
                        handoff_details_by_k[k].append(
                            {
                                "timestamp": ts,
                                "benchmark": base_exec.benchmark,
                                "task_id": base_exec.task_id,
                                "provider": provider,
                                "base_model": model_name,
                                "handoff_models_used": [],
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
                                "patch_explanations": [],
                                "iterations": [],
                                "prompt": task.build_prompt() if hasattr(task, "build_prompt") else "",
                                "handoff_label": handoff_model_label(provider, agents_k),
                            }
                        )
                    continue

                sd = self_debug_stream(
                    task=task,
                    initial_code=candidate,
                    first_exec=base_exec,
                    max_self_debug_iters=max_self_debug_iters,
                    provider=provider,
                    patch_model=model_name,
                )

                print("\n\nFinal code:")
                print(sd["final_code"])

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

                if sd.get("passed") or sd.get("num_tests", 0) == 0:
                    for k in k_values:
                        agents_k = get_patch_agents_for_provider(
                            provider, k, patch_pool, gemini_patch_pool, claude_patch_pool, ollama_patch_pool
                        )
                        handoff_details_by_k[k].append(
                            {
                                "timestamp": ts,
                                "benchmark": base_exec.benchmark,
                                "task_id": base_exec.task_id,
                                "provider": provider,
                                "base_model": model_name,
                                "handoff_models_used": [],
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
                                "patch_explanations": [],
                                "iterations": [],
                                "prompt": task.build_prompt() if hasattr(task, "build_prompt") else "",
                                "skipped_reason": "self_debug_solved",
                                "handoff_label": handoff_model_label(provider, agents_k),
                            }
                        )
                    continue

                for k in k_values:
                    agents_k = get_patch_agents_for_provider(
                        provider, k, patch_pool, gemini_patch_pool, claude_patch_pool, ollama_patch_pool
                    )

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
                            "base_model": model_name,
                            "handoff_models_used": handoff.get("patch_models_used", []),
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
                            "patch_explanations": handoff.get("patch_explanations", []),
                            "iterations": handoff.get("iterations", []),
                            "prompt": task.build_prompt() if hasattr(task, "build_prompt") else "",
                            "handoff_label": handoff_model_label(provider, agents_k),
                        }
                    )

            baseline_summary = summarize_results(baseline_details, benchmark, provider, model_name, "baseline")
            self_debug_summary = summarize_results(self_debug_details, benchmark, provider, model_name, "self_debug_single")

            all_results.append(("baseline", baseline_summary))
            all_results.append(("self_debug_single", self_debug_summary))

            handoff_summaries_by_k: Dict[int, Dict[str, Any]] = {}
            for k in k_values:
                agents_k = get_patch_agents_for_provider(
                    provider, k, patch_pool, gemini_patch_pool, claude_patch_pool, ollama_patch_pool
                )
                tag = f"handoff_{k}agents"
                label = handoff_model_label(provider, agents_k)
                summary_k = summarize_results(handoff_details_by_k[k], benchmark, provider, label, tag)
                all_results.append((tag, summary_k))
                handoff_summaries_by_k[k] = summary_k

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
                "handoff_by_k": {k: handoff_summaries_by_k[k] for k in k_values},
            }
            compact_summary_rows.append(compact_row)

    pass_rate_plot_paths = plot_clean_grouped_bars(all_results, k_values, plots_dir)

    experiment_config = {
        "benchmarks": benchmarks,
        "configs": configs,
        "max_tasks": max_tasks,
        "max_self_debug_iters": max_self_debug_iters,
        "patch_pools": {
            "openai": [{"provider": a.provider, "model": a.model} for a in patch_pool],
            "gemini": [{"provider": a.provider, "model": a.model} for a in gemini_patch_pool],
            "anthropic": [{"provider": a.provider, "model": a.model} for a in claude_patch_pool],
            "ollama": [{"provider": a.provider, "model": a.model} for a in ollama_patch_pool],
        },
        "k_values": k_values,
        "openai_default_model": OPENAI_MODEL,
        "gemini_default_model": GOOGLE_MODEL,
        "anthropic_default_model": CLAUDE_MODEL,
        "ollama_default_model": OLLAMA_MODEL,
    }

    summary_report_text = build_summary_report_text(compact_summary_rows)

    artifacts = {
        "run_dir": run_dir,
        "plots_dir": plots_dir,
        "plots": {
            "pass_rates": pass_rate_plot_paths,
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
        base_model = row["model"]

        bsum = row["baseline"]
        ssum = row["self_debug"]

        print(f"{bench} on {provider}:{base_model}")
        print(f"Baseline: {bsum['num_passed']}/{bsum['num_tasks']} ({bsum['pass_rate']*100:.2f}%)")
        print(f"Self-debug (only on failures(baseline)): {ssum['num_passed']}/{ssum['num_tasks']} ({ssum['pass_rate']*100:.2f}%)")

        for k, hsum in row["handoff_by_k"].items():
            print(f"Handoff ({k}) [{hsum['model']}]: {hsum['num_passed']}/{hsum['num_tasks']} ({hsum['pass_rate']*100:.2f}%)")

        print("")

    print(f"[LOG] Results saved to {saved_path}")
    print("\nEXPERIMENT COMPLETE")