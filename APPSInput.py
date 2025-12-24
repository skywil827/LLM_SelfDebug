from dataclasses import dataclass
from typing import Dict, Any, Optional
from datasets import load_dataset
import random
import json
from datasets.utils.logging import set_verbosity_error


set_verbosity_error()

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

    def build_prompt(self) -> str:
        return self.prompt


def load_apps_task(
    problem_id: Optional[int] = None,
    split: str = "test"
) -> APPSTask:
  
    ds = load_dataset("codeparrot/apps", split=split)

    if problem_id is None:
        sample = ds[random.randrange(len(ds))]
    else:
        matches = [row for row in ds if int(row["problem_id"]) == int(problem_id)]
        if not matches:
            raise ValueError(
                f"APPS problem_id {problem_id} not found in split '{split}'. "
                f"Example available id: {ds[0]['problem_id']}"
            )
        sample = matches[0]

    pid = int(sample["problem_id"])
    question: str = sample["question"]
    starter_code: str = sample.get("starter_code", "") or ""
    difficulty: str = sample.get("difficulty", "unknown")

    io_raw: str = sample.get("input_output", "") or ""
    try:
        io_parsed: Dict[str, Any] = json.loads(io_raw) if io_raw.strip() else {}
    except json.JSONDecodeError:
        io_parsed = {}

    # Build prompt for LLM
    if starter_code.strip():
        full_prompt = (
            question
            + "\n\nYou are given the following starter code. "
              "Complete it to solve the problem.\n"
            "```python\n"
            f"{starter_code}\n"
            "```"
        )
    else:
        full_prompt = (
            question
            + "\n\nWrite a complete Python program that solves this problem."
        )

    constraints = {
        "benchmark": "APPS",
        # "difficulty": difficulty,
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
    }

    return APPSTask(
        problem_id=pid,
        prompt=full_prompt,
        question=question,
        starter_code=starter_code,
        input_output_raw=io_raw,
        input_output=io_parsed,
        difficulty=difficulty,
        constraints=constraints,
    )

# print(load_apps_task())