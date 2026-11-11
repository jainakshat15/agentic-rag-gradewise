# Week 5 evaluation: grader vs human

> **Before Asha trusts GradeWise with real marks, she wants proof. So do we.**

`run_evals.py` compares GradeWise with a human examiner on the 10 synthetic students, and its handwriting transcriptions with what the students really wrote.

```bash
uv run week5/evaluation/run_evals.py                      # evaluate the reports already stored for cs_midterm
uv run week5/evaluation/run_evals.py --grade              # first grade all 10 students from transcripts.json (💰 ~150 calls)
uv run week5/evaluation/run_evals.py --fake               # 🆓 the same with the offline fake grader: a baseline to beat
uv run week5/evaluation/run_evals.py --only transcription # CER / WER only
```

Results print as tables and are saved to `output/evals/grading_eval.json` and `output/evals/transcription_eval.json`.

## The data
| File | What it is |
|---|---|
| `data/golden/synthetic_labels.csv` | The human marks: `student_id,question_id,criterion_id,awarded,max` |
| `data/answer-sheets/synthetic/transcripts.json` | What each student really wrote: `{"student_01": {"Q1": "..."}}` |
| `output/transcriptions/*.json` | What the vision model read (from week 4, or `gradewise transcribe`) |

💡 `--grade` grades from the **true** transcripts, not from the vision output. That isolates the grader: any disagreement is the grader's fault, not the handwriting's. Evaluate the two stages separately, then together.

## The metrics
| Metric | Question it answers | Good |
|---|---|---|
| **MAE** | On average, how many marks apart are we? | < 0.5 |
| **Exact %** | How often do we give the very same mark? | > 60% |
| **Within-1 %** | How often are we at most one mark apart? | > 90% |
| **QWK** (quadratic weighted kappa) | How much do we agree beyond chance, punishing big gaps more than small ones? | ≥ 0.8 |
| **Confusion summary** | Are we too generous or too harsh? (grader higher / lower / same, and a human × grader matrix) | balanced |
| **CER / WER** | How many characters / words did the vision model get wrong? | CER < 5% |

⚠️ QWK is computed on marks rounded **half up** (2.5 → 3). Python's `round()` rounds half to even (2.5 → 2), which would quietly bias the result.

🧪 **TRY:** grade with `--grade`, then again with self-consistency (`grade_batch(..., consistency_runs=3)`). Does QWK improve enough to justify 3× the cost?

🏋️ **Exercise (week5_EXERCISE):** add a *moderation* second grader that only re-marks answers where the first grade is borderline, and measure whether it moves QWK.

The maths is tested on toy numbers you can check by hand: `uv run python -m pytest tests/test_evals.py -q`.
