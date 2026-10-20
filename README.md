# CSCI 3052U — Machine Learning I Group Project

## Project: Early Identification of At-Risk Students

Predicting undergraduate dropout risk (Dropout / Enrolled / Graduate) from enrollment and
early-semester academic data, using the public UCI *"Predict Students' Dropout and Academic
Success"* dataset. See `docs/Research_Proposal.pdf` for the full proposal (task, current
practice, gap, methodology, and success criteria) and `docs/Team_Contract.pdf` for the
team's working agreement.

## Repository layout

```
data/         raw dataset (data.csv, UCI #697, CC BY 4.0)
src/          reusable pipeline code
  data.py           load -> clean -> design matrix -> stratified split -> normalize (M2 decisions)
  models.py         softmax regression with L2 + gradient descent (NumPy, Week 5)
  evaluate.py       macro-F1, accuracy, per-class scores, confusion matrix
  run_baselines.py  regenerates every Milestone 3 result in results/
tests/        sanity checks (split, leakage, normalization, gradient check, determinism)
notebooks/    milestone2_data_approval.ipynb, milestone3_baseline.ipynb (walkthrough)
results/      baseline table, tuning grid, seed runs, sklearn check, figures, experiment log
docs/         proposal, team contract, Milestone 3 technical memo (HTML source + PDF)
```

## Setup

Python 3.13.15 with the pinned packages in `requirements.txt` (the same versions the
Milestone 2 notebook ran on).

```bash
# with uv
uv venv --python 3.13.15 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# or with the standard library
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Data access

`data/data.csv` is the UCI #697 file
(<https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success>,
CC BY 4.0, Realinho et al. 2021). If it's missing, `src/data.py` downloads it again. Either
way, the pipeline checks the file's SHA-256
(`3ef126de5cefff26eb11fbb4237f1a1401cb64b488e2f1d598c23cedeb4c45ae`) and stops if it doesn't
match.

## Reproducing Milestone 3

Run from the repository root, in this order:

```bash
.venv/bin/python -m tests.test_pipeline     # 7 checks, ~15 s
.venv/bin/python -m src.run_baselines       # full run, ~3 min, rewrites results/
```

Then open `notebooks/milestone3_baseline.ipynb` for the walkthrough (it reads `results/`, so
run the script first), or re-execute it:

```bash
cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace milestone3_baseline.ipynb
```

Fixed settings: split seed 42 (official) plus seeds 0–9 for variability, stratified 70/15/15,
softmax regression α = 0.3, λ = 0, 5,000 GD iterations (chosen on validation macro-F1 from the
grid in `results/tuning_grid.csv`). The test set is not scored anywhere in Milestone 3.

### Lab demo (2 minutes, any team member)

1. `.venv/bin/python -m tests.test_pipeline`: shows the split matches M2 and the gradient check passes.
2. `.venv/bin/python -m src.run_baselines --quick`: rebuilds the split from the raw csv and refits the
   baseline in about 4 s, printing the majority / stratified / softmax rows of the table.
3. Open `results/baseline_table.md` and `results/sklearn_check.csv` to explain the discrepancy analysis
   (memo section 4).

## Milestones

| # | Milestone | Due |
|---|---|---|
| 1 | Proposal & Team Contract | Sept 18, 2026 |
| 2 | Dataset & Problem Approval | Oct 2, 2026 |
| 3 | Baseline Reproduction | Oct 19, 2026 |
| 4 | Progress Presentation | Oct 30, 2026 |
| 5 | Final Experiments & Analysis | Nov 20, 2026 |
| 6 | Final Report, Repository & Demo | Dec 4, 2026 |
| 7 | Individual Reflection & Oral Defence | Dec 7, 2026 |

## Status

Milestone 3 (baseline reproduction): pipeline in `src/`, naive benchmarks and a softmax-regression
baseline, baseline table and discrepancy analysis in `results/` and
`docs/Milestone3_Technical_Memo.pdf`. Validation macro-F1 is 0.661 (0.667 ± 0.020 over 10 split
seeds), against 0.222 for the majority class and the pre-registered success bar of 0.272.
