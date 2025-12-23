from dataclasses import dataclass 
from typing import List, Optional, Any, Dict
from datasets import load_dataset
import textwrap
from openai import OpenAI
import json
import random
# from dotenv import load_dotenv
# import os

@dataclass
class SWEInstance:
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

  def short_summary(self) -> str:
    return (
      f"SWE Task {self.instance_id} frm repo {self.repo}\n"
      f"- Base commit: {self.base_commit}\n"
      f"- Created at: {self.created_at} (dataset version: {self.version})\n"
      f"- Fail_To_Pass tests: {len(self.fail_to_pass)}\n"
      f"- Pass_To_Pass tests: {len(self.pass_to_pass)}"
    )

def normalize_to_list(x: Any) -> List[str]:

  if isinstance(x, (list, tuple)):
    return [str(t) for t in x]

  if isinstance(x, str):
    s = x.strip()

    if not s:
      return []
    
    if s.startswith("[") and s.endswith("]"):
      try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
          return [str(t) for t in parsed]
      except json.JSONDecodeError:
        pass
    
    return [s]
  
  return []


def load_swe_instance(index: int | None = None) -> SWEInstance:


  dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

  if index is None:
    index = random.randrange(len(dataset))
  
  row = dataset[index]

  fail_to_pass = normalize_to_list(row.get("FAIL_TO_PASS"))
  pass_to_pass = normalize_to_list(row.get("PASS_TO_PASS"))

  constraints = {
    "benchmark": "SWE-bench_LITE",
    "time_limit_ms": 2000,
    "memory_limit_mb": 256,
  }

  return SWEInstance(
    repo = row["repo"],
    instance_id = row["instance_id"],
    base_commit = row["base_commit"],
    problem_statement = row["problem_statement"],
    hints_text = row.get("hints_text"),
    patch = row["patch"],
    test_patch = row.get("test_patch"),
    created_at = row["created_at"],
    version = row["version"],
    fail_to_pass = fail_to_pass,
    pass_to_pass = pass_to_pass,
    environment_setup_commit = row.get("environment_setup_commit"),
    constraints = constraints,

  )


def build_swe_prompt(task: SWEInstance) -> str:

  hint_block = ""
  if task.hints_text:
    hint_block = f"\nHints from issue / maintainers:\n{task.hints_text}\n"

  fail_tests_preview = ", ".join(task.fail_to_pass[:200]) or "Tests listed in FAIL_TO_PASS"
  pass_tests_preview = ", ".join(task.pass_to_pass[:200]) or "Existing passing tests in PASS_TO_PASS"
  # pass_tests_preview = type(task.pass_to_pass)

  
  prompt = f"""
      You are an expert software engineer working on the repository:

          {task.repo}

      The current codebase is at commit:

          {task.base_commit}

      A bug has been reported with the following problem statement:

      {task.problem_statement}
      {hint_block}

      The test suite is structured so that:
      - FAIL_TO_PASS: tests that currently fail, but should pass after your fix.
        Example / preview: {fail_tests_preview}
      - PASS_TO_PASS: tests that already pass and must continue to pass.
        Example / preview: {pass_tests_preview}

      Your job is to produce a git patch that fixes the bug while preserving existing behavior.

      IMPORTANT REQUIREMENTS:
      - Output must be a unified diff (git patch) that can be applied on top of the base commit.
      - Do NOT include explanations, markdown, or prose in the final answer.
      - Do NOT modify tests unless absolutely necessary.
      - Try to keep the patch as small and focused as possible.

      Return ONLY the patch, in standard unified diff format, starting with lines like:

          diff --git a/path/to/file.py b/path/to/file.py
    """
  
  return textwrap.dedent(prompt).strip()


# print(build_swe_prompt(load_swe_instance()))