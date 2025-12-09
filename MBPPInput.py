from dataclasses import dataclass
from typing import Dict, Any, Optional
from datasets import load_dataset

import random


@dataclass
class MBPPTask:
    task_id: int
    prompt: str
    function_signature: str
    test_code: str
    constraints: Dict[str, Any]

    def build_prompt(self) -> str:
        return self.prompt


def load_mbpp_task(task_id: Optional[int] = None) -> MBPPTask:
    ds = load_dataset("mbpp", "sanitized", split="test")

    if task_id is None:
        sample = ds[random.randrange(len(ds))]
    else:
        matches = [row for row in ds if int(row["task_id"]) == int(task_id)]
        if not matches:
            raise ValueError(
                f"MBPP task_id {task_id} not found. "
                f"Example available id: {ds[0]['task_id']}"
            )
        sample = matches[0]

    tid = int(sample["task_id"])
    nl_prompt: str = sample["prompt"]    
    canonical_code: str = sample["code"]    

   
    sig_line = canonical_code.splitlines()[0].strip()

    starter_code = sig_line + "\n    # TODO: implement\n    pass"
 
    full_prompt = (
        nl_prompt
        + "\n\nYou must implement the following Python function:\n"
        "```python\n"
        f"{starter_code}\n"
        "```"
    )


    imports = "\n".join(sample.get("test_imports", [])) if sample.get("test_imports") else ""
    tests_list = sample.get("test_list", []) or []
    tests_block = "\n".join(tests_list)
    test_code = ((imports + "\n") if imports else "") + tests_block

    constraints = {
        "benchmark": "MBPP",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
    }

    return MBPPTask(
        task_id=tid,
        prompt=full_prompt,
        function_signature=sig_line,
        test_code=test_code,
        constraints=constraints,
    )
