from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
from runtime.code_exec import ExecutionResult
from humanEvalInput import HumanEvalTask
from MBPPInput import MBPPTask
from APPSInput import APPSTask
from sweBenchInput import SWELITETask




@dataclass
class FeedbackPackage:
    benchmark: str
    task_id: str
    passed: bool
    num_tests: int
    num_passed: int
    error_type: Optional[str]
    error_message: Optional[str]
    traceback_excerpt: Optional[str]
    stdout_excerpt: str
    stderr_excerpt: str
    raw_execution: ExecutionResult

    def to_model_feedback_block(self) -> str:
        status = "ALL TESTS PASSED" if self.passed else "TESTS FAILED"
        score_str = (
            f"{self.num_passed}/{self.num_tests} tests passed"
            if self.num_tests > 0
            else "No explicit tests were run"
        )

        lines = [
            f"# Execution Feedback",
            f"- Benchmark: {self.benchmark}",
            f"- Task ID: {self.task_id}",
            f"- Status: {status}",
            f"- Test summary: {score_str}",
            "",
        ]

        if self.error_type or self.error_message:
            lines.append("## Error Summary")
            if self.error_type:
                lines.append(f"- Error type: {self.error_type}")
            if self.error_message:
                lines.append(f"- Error message: {self.error_message}")
            if self.traceback_excerpt:
                lines.append("")
                lines.append("### Traceback (excerpt)")
                lines.append(self.traceback_excerpt)
                lines.append("")
        else:
            lines.append("No Python exception was raised during execution.")
            lines.append("")

        if self.stdout_excerpt.strip():
            lines.append("## STDOUT (excerpt)")
            lines.append(self.stdout_excerpt)
            lines.append("")

        if self.stderr_excerpt.strip():
            lines.append("## STDERR (excerpt)")
            lines.append(self.stderr_excerpt)
            lines.append("")

        return "\n".join(lines)

def _truncate(text: str, max_chars: int = 2000) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n...[truncated]..."

TaskType = Union[HumanEvalTask, MBPPTask, APPSTask, SWELITETask]


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


def build_feedback_package(
    task: TaskType,
    code: str,
    exec_result: ExecutionResult,
    traceback_max_chars: int = 2000,
    stream_max_chars: int = 1000,
) -> FeedbackPackage:
    benchmark, tid = _get_task_identity(task)

    tb_excerpt = _truncate(exec_result.traceback_str or "", traceback_max_chars)
    stdout_excerpt = _truncate(exec_result.stdout or "", stream_max_chars)
    stderr_excerpt = _truncate(exec_result.stderr or "", stream_max_chars)

    return FeedbackPackage(
        benchmark=benchmark,
        task_id=tid,
        passed=exec_result.passed,
        num_tests=exec_result.num_tests,
        num_passed=exec_result.num_passed,
        error_type=exec_result.error_type,
        error_message=exec_result.error_message,
        traceback_excerpt=tb_excerpt,
        stdout_excerpt=stdout_excerpt,
        stderr_excerpt=stderr_excerpt,
        raw_execution=exec_result,
    )
