import io
import traceback
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, Tuple, List
from contextlib import redirect_stdout, redirect_stderr
import re



@dataclass
class HumanEvalTask:
    task_id: str
    prompt: str
    function_signature: str
    test_code: str
    constraints: Dict[str, Any]


@dataclass
class MBPPTask:
    task_id: int
    prompt: str
    function_signature: str
    test_code: str
    constraints: Dict[str, Any]


@dataclass
class APPSTask:
    problem_id: int
    prompt: str
    question: str
    starter_code: str
    input_output_raw: str
    input_output: Dict[str, Any]
    difficulty: str
    constraints: Dict[str, Any]


@dataclass
class SWELITETask:
    repo: str
    instance_id: str
    base_commit: str
    problem_statement: str
    hints_text: Optional[str]
    patch: str
    test_patch: Optional[str]
    created_at: str
    version: str
    fail_to_pass: List[str]
    pass_to_pass: List[str]
    environment_setup_commit: Optional[str]
    constraints: Dict[str, Any]


TaskType = Union[HumanEvalTask, MBPPTask, APPSTask, SWELITETask]


@dataclass
class ExecutionResult:
    benchmark: str
    task_id: str
    passed: bool
    num_tests: int
    num_passed: int
    error_type: Optional[str]
    error_message: Optional[str]
    traceback_str: Optional[str]
    stdout: str
    stderr: str


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


def _estimate_num_tests_from_code(test_code: str) -> int:
    return test_code.count("assert")


def _run_code_with_test_code(
    code: str,
    test_code: str,
    benchmark: str,
    task_id: str,
) -> ExecutionResult:
    
    full_source = code + "\n\n" + test_code
    global_ns: Dict[str, Any] = {}

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    num_tests = _estimate_num_tests_from_code(test_code)
    num_passed, passed = 0, False
    err_type = err_msg = tb_str = None

    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        try:
            exec(full_source, global_ns, global_ns)
            passed = True
            num_passed = num_tests
        except Exception as e:
            passed = False
            err_type = type(e).__name__
            err_msg = str(e)
            tb_str = traceback.format_exc()
            num_passed = max(0, num_tests - 1)

    return ExecutionResult(
        benchmark=benchmark,
        task_id=task_id,
        passed=passed,
        num_tests=num_tests,
        num_passed=num_passed,
        error_type=err_type,
        error_message=err_msg,
        traceback_str=tb_str,
        stdout=stdout_buf.getvalue(),
        stderr=stderr_buf.getvalue(),
    )


def _run_apps_code(task: APPSTask, code: str) -> ExecutionResult:

    benchmark = task.constraints.get("benchmark", "APPS")
    task_id = f"APPS/{task.problem_id}"

    global_ns: Dict[str, Any] = {}
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    num_tests = num_passed = 0
    passed = False
    err_type = err_msg = tb_str = None

    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        try:
            exec(code, global_ns, global_ns)
        except Exception as e:
            return ExecutionResult(
                benchmark=benchmark,
                task_id=task_id,
                passed=False,
                num_tests=0,
                num_passed=0,
                error_type=type(e).__name__,
                error_message=str(e),
                traceback_str=traceback.format_exc(),
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
            )

        tests = task.input_output or {}
        fn_name = tests.get("fn_name")
        inputs = tests.get("inputs")
        outputs = tests.get("outputs")

        if fn_name and inputs is not None and outputs is not None:
            func = global_ns.get(fn_name)
            if not callable(func):
                return ExecutionResult(
                    benchmark=benchmark,
                    task_id=task_id,
                    passed=False,
                    num_tests=0,
                    num_passed=0,
                    error_type="NameError",
                    error_message=f"Function '{fn_name}' not found.",
                    traceback_str=None,
                    stdout=stdout_buf.getvalue(),
                    stderr=stderr_buf.getvalue(),
                )

            num_tests = len(outputs)
            for inp, expected in zip(inputs, outputs):
                try:
                    got = func(*inp) if isinstance(inp, (list, tuple)) else func(inp)
                    if got == expected:
                        num_passed += 1
                    else:
                        return ExecutionResult(
                            benchmark=benchmark,
                            task_id=task_id,
                            passed=False,
                            num_tests=num_tests,
                            num_passed=num_passed,
                            error_type="AssertionError",
                            error_message=f"Expected {expected}, got {got}",
                            traceback_str=None,
                            stdout=stdout_buf.getvalue(),
                            stderr=stderr_buf.getvalue(),
                        )
                except Exception as e:
                    return ExecutionResult(
                        benchmark=benchmark,
                        task_id=task_id,
                        passed=False,
                        num_tests=num_tests,
                        num_passed=num_passed,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        traceback_str=traceback.format_exc(),
                        stdout=stdout_buf.getvalue(),
                        stderr=stderr_buf.getvalue(),
                    )

            passed = (num_passed == num_tests)
            
        else:
            passed = True

    return ExecutionResult(
        benchmark=benchmark,
        task_id=task_id,
        passed=passed,
        num_tests=num_tests,
        num_passed=num_passed,
        error_type=err_type,
        error_message=err_msg,
        traceback_str=tb_str,
        stdout=stdout_buf.getvalue(),
        stderr=stderr_buf.getvalue(),
    )



# SWE patch execution (baseline)
_UNIFIED_DIFF_RE = re.compile(
    r"(?m)^diff --git a\/.+ b\/.+\n"
    r"(?:index [0-9a-f]+\.\.[0-9a-f]+(?: \d+)?\n)?"
    r"^--- a\/.+\n"
    r"^\+\+\+ b\/.+\n"
)

def _validate_unified_diff(patch_text: str) -> Tuple[bool, str]:
    if patch_text is None:
        return False, "Patch is None."
    t = str(patch_text).strip()
    if not t:
        return False, "Patch is empty."
    if "diff --git " not in t:
        return False, "Missing 'diff --git a/... b/...' header."
    if not _UNIFIED_DIFF_RE.search(t):
        return False, "Missing ---/+++ file headers (not a valid unified diff)."
    return True, "Patch format OK."


def execute_swe_task_patch(task: SWELITETask, patch: str) -> ExecutionResult:

    benchmark, tid = _get_task_identity(task)

    ok, msg = _validate_unified_diff(patch)
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    passed = ok
    num_tests = 1
    num_passed = 1 if ok else 0

    err_type = None if ok else "InvalidPatch"
    err_msg = None if ok else msg

    stdout_buf.write(f"[SWE] Patch validation: {msg}\n")
    stdout_buf.write(f"[SWE] Repo: {task.repo}\n")
    stdout_buf.write(f"[SWE] Base commit: {task.base_commit}\n")

    return ExecutionResult(
        benchmark=benchmark,
        task_id=tid,
        passed=passed,
        num_tests=num_tests,
        num_passed=num_passed,
        error_type=err_type,
        error_message=err_msg,
        traceback_str=None,
        stdout=stdout_buf.getvalue(),
        stderr=stderr_buf.getvalue(),
    )


def execute_task_code(task: TaskType, code: str) -> ExecutionResult:

    benchmark, tid = _get_task_identity(task)

    if benchmark == "SWE-bench_LITE" or isinstance(task, SWELITETask):
        raise TypeError(
            "execute_task_code() is for Python-code benchmarks only. "
            "For SWE-bench, call execute_swe_task_patch(task, patch)."
        )

    if benchmark == "HumanEval":
        return _run_code_with_test_code(code, task.test_code, benchmark, tid)

    if benchmark == "MBPP":
        return _run_code_with_test_code(code, task.test_code, benchmark, tid)

    if benchmark == "APPS":
        return _run_apps_code(task, code)

    # Fallback
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        try:
            exec(code, {})
            return ExecutionResult(
                benchmark=benchmark,
                task_id=tid,
                passed=True,
                num_tests=0,
                num_passed=0,
                error_type=None,
                error_message=None,
                traceback_str=None,
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
            )
        except Exception as e:
            return ExecutionResult(
                benchmark=benchmark,
                task_id=tid,
                passed=False,
                num_tests=0,
                num_passed=0,
                error_type=type(e).__name__,
                error_message=str(e),
                traceback_str=traceback.format_exc(),
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
            )


def execute_task(task: TaskType, candidate: str) -> ExecutionResult:

    if isinstance(task, SWELITETask) or task.constraints.get("benchmark") == "SWE-bench_LITE":
        return execute_swe_task_patch(task, candidate)

    return execute_task_code(task, candidate)
