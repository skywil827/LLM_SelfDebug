from dataclasses import dataclass
from typing import Dict, Any
from human_eval.data import read_problems
import random


@dataclass
class HumanEvalTask:
    task_id: str
    prompt: str
    function_signature: str
    test_code: str
    constraints: Dict[str, Any]

    def build_prompt(self) -> str:
        return self.prompt


def load_humaneval_task(task_id: str | None = None) -> HumanEvalTask:

    problems_dict = read_problems() 

    if task_id is None:
        tid = random.choice(list(problems_dict.keys()))
    else:
        tid = task_id
        if tid not in problems_dict:
            raise ValueError(
                f"HumanEval task_id '{tid}' not found. "
                f"Example available: {next(iter(problems_dict.keys()))}"
            )

    problem = problems_dict[tid]

    full_prompt: str = problem["prompt"]
    test_code: str = problem["test"]

    first_line = full_prompt.splitlines()[0].strip()

    constraints = {
        "benchmark": "HumanEval",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
    }

    return HumanEvalTask(
        task_id=tid,
        prompt=full_prompt,
        function_signature=first_line,
        test_code=test_code,
        constraints=constraints,
    )
