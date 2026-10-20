# AI Use Statement

Required by the project rules (sections 9–10). One entry per milestone.

## Milestone 2
- AI was used to improve some of the descriptions and summaries of the code (see the end of
  `notebooks/milestone2_data_approval.ipynb`).

## Milestone 3

| | |
|---|---|
| Tool | Claude Code (Anthropic) |
| Purpose | Drafting the `src/` modules (refactoring the M2 notebook into functions; the NumPy softmax regression), `tests/test_pipeline.py`, `notebooks/milestone3_baseline.ipynb`, the README update and the technical memo |
| Material given to the tool | Our own repository (M2 notebook, proposal, course project instructions), our lab notebooks and lecture slides for style and theory, and the public UCI dataset (CC BY 4.0, no personal identifiers). No confidential or partner data |
| Sources | The published-result comparison uses only the Realinho et al. (2022) data descriptor, which we read. No numbers from Martins et al. (2021) are reported, because we couldn't read that paper |

**How the output is verified**
- Every number in the memo, notebook and `results/` comes from running `python -m src.run_baselines`.
  Two full runs gave identical results, and `results/experiment_log.csv` records each run.
- The analytic gradient of the softmax cost is compared with finite differences (`tests/`).
- The NumPy model is compared with an independent library fit (scikit-learn `LogisticRegression`) at
  every λ in the grid (`results/sklearn_check.csv`).
- The pipeline reproduces the numbers we recorded in M2 by hand: 169 features, a 3,096 / 664 / 664
  split, and majority macro-F1 0.222.

**Before submission, every member must:** read `src/data.py`, `src/models.py` and
`src/run_baselines.py`; be able to derive the softmax gradient (y − t)x and explain the λ = 0
discrepancy (memo section 4); and confirm this statement is accurate.
