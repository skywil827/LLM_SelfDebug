This repository evaluates a self-debugging pipeline for code-generating Large Language Models across three benchmark datasets:
  HumanEval
  MBPP
  APPS
Each experiment compares Baseline performance against the self-debugging performance using iterative error analysis and code pathing.

NOTE: To manage computational cost, the full benchmark datasets were not executed. Each model was tested on a representative sample of 30 tasks, 
and the self-debugging module was allowed to perform up to 10 correction iterations before termination.


RESULTS:
(tvenv) osborn@OSBORN-PC:~/thesis$ python3.11 ./index.py

HumanEval on openai:gpt-4o | BASELINE
Baseline: 29/30 (96.67% pass rate)
HumanEval on openai:gpt-4o | SELF-DEBUG
Self-debug: 30/30 (100.00% pass rate)
Improvement: +1 tasks, +3.3333 percentage points

MBPP on openai:gpt-4o | BASELINE
Baseline: 24/30 (80.00% pass rate)
MBPP on openai:gpt-4o | SELF-DEBUG
Self-debug: 26/30 (86.67% pass rate)
Improvement: +2 tasks, +6.6667 percentage points

HumanEval on openai:gpt-4o-mini | BASELINE
Baseline: 21/30 (70.00% pass rate)
HumanEval on openai:gpt-4o-mini | SELF-DEBUG
Self-debug: 30/30 (100.00% pass rate)
Improvement: +9 tasks, +30.0000 percentage points

MBPP on openai:gpt-4o-mini | BASELINE
Baseline: 22/30 (73.33% pass rate)
MBPP on openai:gpt-4o-mini | SELF-DEBUG
Self-debug: 25/30 (83.33% pass rate)
Improvement: +3 tasks, +10.0000 percentage points

HumanEval on openai:gpt-5.1 | BASELINE
Baseline: 30/30 (100.00% pass rate)
HumanEval on openai:gpt-5.1 | SELF-DEBUG
Self-debug: 30/30 (100.00% pass rate)
Improvement: +0 tasks, +0.0000 percentage points

MBPP on openai:gpt-5.1 | BASELINE
Baseline: 25/30 (83.33% pass rate)
MBPP on openai:gpt-5.1 | SELF-DEBUG
Self-debug: 26/30 (86.67% pass rate)
Improvement: +1 tasks, +3.3333 percentage points

HumanEval on gemini:gemini-2.0-flash | BASELINE
Baseline: 30/30 (100.00% pass rate)
HumanEval on gemini:gemini-2.0-flash | SELF-DEBUG
Self-debug: 30/30 (100.00% pass rate)
Improvement: +0 tasks, +0.0000 percentage points

MBPP on gemini:gemini-2.0-flash | BASELINE
Baseline: 0/30 (0.00% pass rate)
MBPP on gemini:gemini-2.0-flash | SELF-DEBUG
Self-debug: 28/30 (93.33% pass rate)
Improvement: +28 tasks, +93.3333 percentage points

HumanEval on gemini:gemini-2.5-pro | BASELINE
Baseline: 30/30 (100.00% pass rate)
HumanEval on gemini:gemini-2.5-pro | SELF-DEBUG
Self-debug: 30/30 (100.00% pass rate)
Improvement: +0 tasks, +0.0000 percentage points

MBPP on gemini:gemini-2.5-pro | BASELINE
Baseline: 0/30 (0.00% pass rate)
MBPP on gemini:gemini-2.5-pro | SELF-DEBUG
Self-debug: 28/30 (93.33% pass rate)
Improvement: +28 tasks, +93.3333 percentage points
