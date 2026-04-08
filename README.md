# LLM Self-Debugging Pipeline Evaluation

This repository evaluates a self-debugging pipeline for code-generating Large Language Models across three benchmark datasets:
  HumanEval
  MBPP
  APPS
Each experiment compares Baseline performance against the self-debugging performance using iterative error analysis and code pathing.

### NOTE: To manage computational cost, I did not run the full benchmark datasets. I tested each model on a  sample of 30 tasks, and set the self-debugging module to perform for up to 10 correction iterations before termination.

```
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
```

**Performance Summary Across Benchmarks**

The results below show how self-debugging and handoff strategies improve code generation across all the benchmarks, such as SWE-bench LITE, HumanEval, MBPP, and APPS, using these frontier models like GPT-5.4, Claude Opus 4.6, and Gemini 3.1 Pro Preview.

I ran several tests with different dataset sizes to make sure the results are reliable and not due to chance. For example, I tested GPT-5.4 and Claude Opus 4.6 on different numbers of tasks (50, 100, 150, 200, and 300) to show that the results are consistent. Larger runs help confirm overall performance, while smaller runs help show how stable the model is across different sets of problems. This is important because it shows that self-debugging works well in my proposed framework, and smaller runs still help explain the results clearly.

The results show that baseline performance is different across benchmarks and models, ranging from moderate accuracy (around 79% on MBPP) to very high or perfect accuracy (100% on HumanEval and APPS in some cases). When the baseline is not perfect, self-debugging helps improve performance by fixing mistakes. For example, on MBPP using GPT-5.4, performance improved from 79% to 89%, and then to 91% with handoff. Also, on SWE-bench LITE, GPT-5.4 improved from 89.67% at baseline to 100% after self-debugging. Claude Opus 4.6 also showed steady improvement, such as increasing from 88.67% to above 94% on MBPP after applying debugging and handoff.

Sequential handoff allows multiple models to work on the same task one after another. If a model cannot fully solve a task after a certain number of attempts, the task is passed to another step for further improvement. The results show that handoff gives extra improvement after self-debugging and often helps reach 100%, especially on SWE-bench LITE and MBPP. However, the improvement from handoff is usually smaller compared to the improvement from baseline to self-debugging. In many cases, performance stops improving once it gets close to perfect, so running more iterations does not help much. 

When the baseline performance is already very high or perfect, such as in many HumanEval and APPS tests, both self-debugging and handoff do not make much difference. This is because there are very few or no errors left to fix.

```
Model:  HumanEval Results (GPT-5.4)
Tasks	Baseline	Self-Debug	Handoff (2)	Handoff (3)
50	50/50 (100%)	50/50 (100%)	50/50 (100%)	50/50 (100%)
100	99/100 (99%)	100/100 (100%)	100/100 (100%)	100/100 (100%)
150	150/150 (100%)	150/150 (100%)	150/150 (100%)	150/150 (100%)
164	164/164 (100%)	164/164 (100%)	164/164 (100%)	164/164 (100%)

Model: MBPP Results (GPT-5.4)
Tasks	Baseline	Self-Debug	Handoff (2)	Handoff (3)
50	42/50 (84%)	49/50 (98%)	49/50 (98%)	49/50 (98%)
100	79/100 (79%)	89/100 (89%)	91/100 (91%)	91/100 (91%)

Model: APPS Results (GPT-5.4)
Tasks	Baseline	Self-Debug	Handoff (2)	Handoff (3)
50	50/50 (100%)	50/50 (100%)	50/50 (100%)	50/50 (100%)
100	97/100 (97%)	97/100 (97%)	97/100 (97%)	97/100 (97%)

Model: SWE-bench LITE (GPT-5.4)
Tasks	Baseline	Self-Debug	Handoff (2)	Handoff (3)
50	49/50 (98%)	50/50 (100%)	50/50 (100%)	50/50 (100%)
100	95/100 (95%)	100/100 (100%)	100/100 (100%)	100/100 (100%)
300	269/300 (89.67%)	300/300 (100%)	300/300 (100%)	300/300 (100%)


Model: SWE-bench LITE (Claude Opus 4.6)
Tasks	Baseline	Self-Debug	Handoff (2)	Handoff (3)
50	47/50 (94%)	49/50 (98%)	50/50 (100%)	50/50 (100%)
100	95/100 (95%)	96/100 (96%)	100/100 (100%)	100/100 (100%)
150	141/150 (94%)	146/150 (97.33%)	150/150 (100%)	150/150 (100%)

Model: MBPP (Claude Opus 4.6)
Tasks	Baseline	Self-Debug	Handoff (2)	Handoff (3)
50	46/50 (92%)	50/50 (100%)	50/50 (100%)	50/50 (100%)
100	88/100 (88%)	94/100 (94%)	94/100 (94%)	95/100 (95%)
150	133/150 (88.67%)	141/150 (94%)	142/150 (94.67%)	141/150 (94%)
200	167/200 (83.50%)	184/200 (92%)	187/200 (93.50%)	187/200 (93.50%)

Model: HumanEval (Claude Opus 4.6)
Tasks	Baseline	Self-Debug	Handoff (2)	Handoff (3)
50	50/50 (100%)	50/50 (100%)	50/50 (100%)	50/50 (100%)
100	100/100 (100%)	100/100 (100%)	100/100 (100%)	100/100 (100%)
150	150/150 (100%)	150/150 (100%)	150/150 (100%)	150/150 (100%)

Model: HumanEval (Gemini 3.1 Pro Preview)
Tasks	Baseline	Self-Debug	Handoff (2)	Handoff (3)
50	46/50 (92%)	50/50 (100%)	50/50 (100%)	50/50 (100%)
100	96/100 (96%)	100/100 (100%)	100/100 (100%)	100/100 (100%)
```
