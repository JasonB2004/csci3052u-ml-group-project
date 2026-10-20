"""
Milestone 3: regenerate every baseline result from the raw csv.

    python -m src.run_baselines          # full run, ~3 min, writes results/
    python -m src.run_baselines --quick  # lab demo, ~15 s, prints the table, writes nothing

Writes everything to results/. The test set is never scored here: it's used once,
at the end of the project (Week 3 p.51-52, M2 section 8), so only its size is recorded.
"""
import csv
import json
import platform
import sys
import time
import warnings
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.data import (CLASS_ORDER, DATA_PATH, DATA_SHA256, RANDOM_STATE, REPO_ROOT,
                      file_sha256, load_splits)
from src.evaluate import evaluate, plot_confusion
from src.models import SoftmaxRegression

RESULTS_DIR = REPO_ROOT / "results"

# search space (W4 p.90: grid search the learning rate, try 0.1, 0.03, 0.01, ...;
# W4 p.76: lambda is a hyperparameter we tune with a validation set)
ALPHAS = [0.3, 0.1, 0.03, 0.01]
LAMBDAS = [0, 0.001, 0.01, 0.1, 1]
N_ITERS = 5000

# extra split seeds for variability (rubric section 6: report variability, not one run)
SEEDS = list(range(10))

# numbers recorded in the Milestone 2 notebook, for the discrepancy table
M2_RECORDED = {"majority_valid_macro_f1": 0.222, "n_features": 169,
               "split_sizes": [3096, 664, 664], "success_bar": 0.272}


def build_all_models(alphas, lambdas, X_train, t_train, X_valid, t_valid, n_iters=N_ITERS):
    """
    Fit one softmax regression per (alpha, lambda) pair (same idea as build_all_models in Lab 4).

    Parameters:
        `alphas` - learning rates to try
        `lambdas` - L2 strengths to try
        `X_train`, `t_train` - training data the models learn from
        `X_valid`, `t_valid` - validation data used to compare them
        `n_iters` - gradient descent steps for every model
    Returns: a dictionary `out` keyed by (alpha, lambda), where
             out[(a, lam)]['val'] = validation macro-F1,
             out[(a, lam)]['train'] = training macro-F1,
             out[(a, lam)]['J'] = final training cost
    """
    out = {}
    for a in alphas:
        for lam in lambdas:
            model = SoftmaxRegression(alpha=a, lam=lam, n_iters=n_iters).fit(X_train, t_train)
            out[(a, lam)] = {
                "val": evaluate(t_valid, model.predict(X_valid))["macro_f1"],
                "train": evaluate(t_train, model.predict(X_train))["macro_f1"],
                "J": model.cost_history[-1],
            }
    return out


def sklearn_logreg(lam, n_train):
    """
    The library version of the same model, used only to check our NumPy implementation.

    sklearn minimizes 0.5*||w||^2 + C * sum_i CE_i, while we minimize
    (1/N) sum_i CE_i + (lambda/2)||w||^2. Multiplying ours by N/lambda gives theirs
    with C = 1 / (lambda * N). Neither penalizes the bias.

    Parameters:
        `lam` - our lambda
        `n_train` - number of training examples N
    Returns: an unfitted LogisticRegression
    """
    if lam == 0:
        # with no penalty the one-hot columns let some weights grow forever, so lbfgs
        # never converges (it ran 10,000 iterations / 200 s without stopping); cap it.
        # sklearn_check.csv records that it didn't converge, so the warning is silenced
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        return LogisticRegression(penalty=None, max_iter=1000, tol=1e-8)
    return LogisticRegression(C=1 / (lam * n_train), max_iter=10000, tol=1e-8)


def row(name, split_seed, t_train, y_train, t_valid, y_valid, notes=""):
    """
    Parameters:
        `name` - model name for the table
        `split_seed` - which split was used
        `t_train`, `y_train` - training labels / predictions
        `t_valid`, `y_valid` - validation labels / predictions
        `notes` - free text
    Returns: one dict = one row of the baseline table
    """
    tr = evaluate(t_train, y_train)
    va = evaluate(t_valid, y_valid)
    return {"model": name, "split_seed": split_seed,
            "train_macro_f1": tr["macro_f1"], "valid_macro_f1": va["macro_f1"],
            "valid_accuracy": va["accuracy"],
            "valid_recall_Dropout": va["recall_Dropout"],
            "valid_recall_Enrolled": va["recall_Enrolled"],
            "valid_recall_Graduate": va["recall_Graduate"],
            "valid_f1_Dropout": va["f1_Dropout"],
            "valid_f1_Enrolled": va["f1_Enrolled"],
            "valid_f1_Graduate": va["f1_Graduate"],
            "notes": notes}


def run_naive(s, seed):
    """
    Parameters:
        `s` - splits from load_splits()
        `seed` - random_state for the stratified dummy
    Returns: (majority row, stratified-random row)
    """
    majority = DummyClassifier(strategy="most_frequent").fit(s["X_train"], s["t_train"])
    stratified = DummyClassifier(strategy="stratified", random_state=seed).fit(s["X_train"], s["t_train"])
    return (row("Majority class", seed, s["t_train"], majority.predict(s["X_train"]),
                s["t_valid"], majority.predict(s["X_valid"]), "always predicts Graduate"),
            row("Stratified random", seed, s["t_train"], stratified.predict(s["X_train"]),
                s["t_valid"], stratified.predict(s["X_valid"]), "random labels with training class rates"))


def quick_demo():
    """
    Lab demonstration: rebuild the official split and refit the chosen baseline
    (hyperparameters read from results/discrepancy.json), no tuning, no files written.
    """
    with open(RESULTS_DIR / "discrepancy.json") as f:
        chosen = json.load(f)["chosen_hyperparameters"]
    s = load_splits(seed=RANDOM_STATE)
    print("Design matrix:", s["X"].shape, "| train/valid/test:",
          len(s["X_train"]), len(s["X_valid"]), len(s["X_test"]))
    majority_row, stratified_row = run_naive(s, RANDOM_STATE)
    model = SoftmaxRegression(alpha=chosen["alpha"], lam=chosen["lambda"],
                              n_iters=chosen["n_iters"]).fit(s["X_train"], s["t_train"])
    softmax_row = row("Softmax regression (NumPy GD)", RANDOM_STATE, s["t_train"], model.predict(s["X_train"]),
                      s["t_valid"], model.predict(s["X_valid"]))
    table = pd.DataFrame([majority_row, stratified_row, softmax_row])
    print(table[["model", "train_macro_f1", "valid_macro_f1", "valid_accuracy",
                 "valid_recall_Dropout", "valid_recall_Enrolled", "valid_recall_Graduate"]]
          .round(3).to_string(index=False))


def main():
    if "--quick" in sys.argv:
        return quick_demo()
    start = time.perf_counter()
    RESULTS_DIR.mkdir(exist_ok=True)

    # 1. the official split (seed 42), exactly as in Milestone 2
    s = load_splits(seed=RANDOM_STATE)
    X_train, X_valid, t_train, t_valid = s["X_train"], s["X_valid"], s["t_train"], s["t_valid"]
    print("Design matrix:", s["X"].shape, "| train/valid/test:",
          len(X_train), len(X_valid), len(s["X_test"]))

    # save the split row indices so anyone can rebuild the same split
    with open(RESULTS_DIR / "split_indices.json", "w") as f:
        json.dump({"dataset_sha256": DATA_SHA256, "random_state": RANDOM_STATE,
                   "method": "sklearn train_test_split, stratified on Target, test_size=0.3 then 0.5",
                   "train": sorted(X_train.index.tolist()),
                   "valid": sorted(X_valid.index.tolist()),
                   "test": sorted(s["X_test"].index.tolist())}, f)

    # 2. naive benchmarks
    majority_row, stratified_row = run_naive(s, RANDOM_STATE)
    print(f"Majority class      valid macro-F1: {majority_row['valid_macro_f1']:.4f}")
    print(f"Stratified random   valid macro-F1: {stratified_row['valid_macro_f1']:.4f}")

    # 3. tune softmax regression on validation only (Lab 4 style search)
    res = build_all_models(ALPHAS, LAMBDAS, X_train, t_train, X_valid, t_valid)
    pd.DataFrame([{"alpha": a, "lambda": lam, "n_iters": N_ITERS, "train_macro_f1": v["train"],
                   "valid_macro_f1": v["val"], "final_J": v["J"]} for (a, lam), v in res.items()]
                 ).to_csv(RESULTS_DIR / "tuning_grid.csv", index=False)

    optimal_alpha = 0
    optimal_lambda = 0
    highest_validation_f1_so_far = 0
    for (a, lam) in res:
        if res[(a, lam)]["val"] > highest_validation_f1_so_far:
            highest_validation_f1_so_far = res[(a, lam)]["val"]
            optimal_alpha = a
            optimal_lambda = lam
    print(f"Optimal alpha: {optimal_alpha}, optimal lambda: {optimal_lambda}, "
          f"validation macro-F1: {highest_validation_f1_so_far:.4f}")

    best = SoftmaxRegression(alpha=optimal_alpha, lam=optimal_lambda, n_iters=N_ITERS).fit(X_train, t_train)
    y_valid_best = best.predict(X_valid)
    softmax_row = row("Softmax regression (NumPy GD)", RANDOM_STATE, t_train, best.predict(X_train),
                      t_valid, y_valid_best, f"alpha={optimal_alpha}, lambda={optimal_lambda}, {N_ITERS} iters")

    # 4. check against sklearn at the same lambda (the CE loss is convex, W5 p.79, so a
    # converged GD should land on the same answer)
    sk = sklearn_logreg(optimal_lambda, len(X_train)).fit(X_train, t_train)
    y_valid_sk = sk.predict(X_valid)
    agreement = float((y_valid_sk == y_valid_best).mean())
    sklearn_row = row("sklearn LogisticRegression (check)", RANDOM_STATE, t_train, sk.predict(X_train),
                      t_valid, y_valid_sk, f"same lambda, lbfgs; agrees with NumPy on {agreement:.1%} of valid")

    # the same check at every lambda > 0 in the grid, where the regularized cost has one
    # finite minimum and both optimizers should reach it. J for sklearn is computed with
    # our own cost function so the two numbers are comparable
    X1 = np.hstack([np.ones((len(X_train), 1)), X_train.to_numpy(dtype=float)])
    T = np.eye(3)[t_train.to_numpy()]
    check_rows = []
    for lam in sorted(set([optimal_lambda] + [l for l in LAMBDAS if l > 0])):
        ours = best if lam == optimal_lambda else \
            SoftmaxRegression(alpha=optimal_alpha, lam=lam, n_iters=N_ITERS).fit(X_train, t_train)
        lib = sk if lam == optimal_lambda else sklearn_logreg(lam, len(X_train)).fit(X_train, t_train)
        lib_as_ours = SoftmaxRegression(lam=lam)
        lib_as_ours.W = np.vstack([lib.intercept_, lib.coef_.T])
        check_rows.append({
            "lambda": lam, "sklearn_C": "none (penalty=None)" if lam == 0 else 1 / (lam * len(X_train)),
            "sklearn_iterations": int(lib.n_iter_[0]), "sklearn_converged": bool(lib.n_iter_[0] < lib.max_iter),
            "final_J_numpy": ours.cost_history[-1], "final_J_sklearn": lib_as_ours.cost(X1, T),
            "valid_macro_f1_numpy": evaluate(t_valid, ours.predict(X_valid))["macro_f1"],
            "valid_macro_f1_sklearn": evaluate(t_valid, lib.predict(X_valid))["macro_f1"],
            "valid_prediction_agreement": float((ours.predict(X_valid) == lib.predict(X_valid)).mean()),
            "max_abs_weight_diff": float(np.abs(ours.W - lib_as_ours.W).max()),
        })
    check_df = pd.DataFrame(check_rows)
    check_df.to_csv(RESULTS_DIR / "sklearn_check.csv", index=False)
    print(check_df.round(4).to_string(index=False))
    J_numpy = check_df.loc[check_df["lambda"] == optimal_lambda, "final_J_numpy"].item()
    J_sklearn = check_df.loc[check_df["lambda"] == optimal_lambda, "final_J_sklearn"].item()

    # 5. variability: same tuned config on 10 other splits (+ naive benchmarks on each)
    seed_rows = []
    for seed in SEEDS:
        ss = load_splits(seed=seed)
        maj, strat = run_naive(ss, seed)
        m = SoftmaxRegression(alpha=optimal_alpha, lam=optimal_lambda, n_iters=N_ITERS).fit(ss["X_train"], ss["t_train"])
        seed_rows += [maj, strat, row("Softmax regression (NumPy GD)", seed, ss["t_train"], m.predict(ss["X_train"]),
                                      ss["t_valid"], m.predict(ss["X_valid"]))]
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(RESULTS_DIR / "seed_runs.csv", index=False)
    seed_summary = seed_df.groupby("model")["valid_macro_f1"].agg(["mean", "std"])
    print(seed_summary.round(4))

    # 6. sensitivity: add the 2nd-semester columns back (the published dataset setup,
    # but outside our end-of-1st-semester prediction point)
    s2 = load_splits(seed=RANDOM_STATE, keep_second_sem=True)
    m2 = SoftmaxRegression(alpha=optimal_alpha, lam=optimal_lambda, n_iters=N_ITERS).fit(s2["X_train"], s2["t_train"])
    second_sem_row = row("Softmax regression + 2nd-sem features", RANDOM_STATE, s2["t_train"], m2.predict(s2["X_train"]),
                         s2["t_valid"], m2.predict(s2["X_valid"]),
                         f"sensitivity only: {s2['X'].shape[1]} features, uses data after our prediction point")

    # 7. baseline table: official split + mean/std over the 10 extra seeds
    table = pd.DataFrame([majority_row, stratified_row, softmax_row, sklearn_row, second_sem_row])
    table["valid_macro_f1_mean_10_seeds"] = table["model"].map(seed_summary["mean"])
    table["valid_macro_f1_std_10_seeds"] = table["model"].map(seed_summary["std"])
    table.to_csv(RESULTS_DIR / "baseline_table.csv", index=False)

    show = table[["model", "train_macro_f1", "valid_macro_f1", "valid_macro_f1_mean_10_seeds",
                  "valid_macro_f1_std_10_seeds", "valid_accuracy", "valid_recall_Dropout",
                  "valid_recall_Enrolled", "valid_recall_Graduate"]].round(3)
    show = show.astype(object).where(show.notna(), "-")   # the check/sensitivity rows have no seed runs
    with open(RESULTS_DIR / "baseline_table.md", "w") as f:
        f.write("Validation results, official split (seed 42). Mean/std are over split seeds 0-9. "
                "Test set not used.\n\n")
        f.write(_markdown_table(show))
        f.write("\n")
    print(show.to_string(index=False))

    # 8. figures
    # training curve (W4 p.90: diagnose optimization with a training curve)
    plt.figure(figsize=(6, 3.5))
    for a in ALPHAS:
        m = SoftmaxRegression(alpha=a, lam=optimal_lambda, n_iters=N_ITERS).fit(X_train, t_train)
        plt.plot(m.cost_history, label=f"alpha = {a}")
    plt.title(f"Training curve, lambda = {optimal_lambda}")
    plt.xlabel("Gradient descent iteration"); plt.ylabel("Training cost J"); plt.legend()
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "training_curve.png", dpi=150); plt.close()

    # train vs valid macro-F1 across lambda (Lab 3 train/valid curve)
    plt.figure(figsize=(6, 3.5))
    plt.plot([str(l) for l in LAMBDAS], [res[(optimal_alpha, l)]["train"] for l in LAMBDAS], marker="o", label="Train")
    plt.plot([str(l) for l in LAMBDAS], [res[(optimal_alpha, l)]["val"] for l in LAMBDAS], marker="o", label="Valid")
    plt.axhline(majority_row["valid_macro_f1"], color="grey", linestyle="--", label="Majority class (valid)")
    plt.title(f"Macro-F1 vs L2 strength, alpha = {optimal_alpha}")
    plt.xlabel("lambda"); plt.ylabel("Macro-F1"); plt.legend()
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "tuning_curve.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(4.5, 4))
    plot_confusion(t_valid, y_valid_best, "Softmax regression, validation", ax=ax)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "confusion_softmax.png", dpi=150); plt.close()

    # 9. discrepancy numbers + environment + experiment log
    p = t_train.value_counts(normalize=True).sort_index()
    p_grad = p[2]
    discrepancy = {
        "majority_valid_macro_f1": {"m2_recorded": M2_RECORDED["majority_valid_macro_f1"],
                                    "analytic": (2 * p_grad / (1 + p_grad)) / 3,
                                    "reproduced": majority_row["valid_macro_f1"]},
        "stratified_valid_macro_f1": {"analytic_approx": float(p.mean()),
                                      "reproduced_seed42": stratified_row["valid_macro_f1"],
                                      "reproduced_mean_10_seeds": float(seed_summary.loc["Stratified random", "mean"]),
                                      "reproduced_std_10_seeds": float(seed_summary.loc["Stratified random", "std"])},
        "n_features": {"m2_recorded": M2_RECORDED["n_features"], "reproduced": int(s["X"].shape[1])},
        "split_sizes": {"m2_recorded": M2_RECORDED["split_sizes"],
                        "reproduced": [len(X_train), len(X_valid), len(s["X_test"])]},
        "numpy_vs_sklearn": {"chosen_lambda": optimal_lambda,
                             "sklearn_converged_at_chosen_lambda": bool(sk.n_iter_[0] < sk.max_iter),
                             "valid_prediction_agreement": agreement,
                             "valid_macro_f1_numpy": softmax_row["valid_macro_f1"],
                             "valid_macro_f1_sklearn": sklearn_row["valid_macro_f1"],
                             "final_J_numpy": float(J_numpy), "final_J_sklearn": float(J_sklearn),
                             "per_lambda": "see sklearn_check.csv"},
        "success_bar": {"m2_preregistered": M2_RECORDED["success_bar"],
                        "softmax_valid_macro_f1": softmax_row["valid_macro_f1"],
                        "met": bool(softmax_row["valid_macro_f1"] >= M2_RECORDED["success_bar"])},
        "second_sem_sensitivity": {"valid_macro_f1_1st_sem_only": softmax_row["valid_macro_f1"],
                                   "valid_macro_f1_with_2nd_sem": second_sem_row["valid_macro_f1"]},
        "chosen_hyperparameters": {"alpha": optimal_alpha, "lambda": optimal_lambda, "n_iters": N_ITERS},
    }
    with open(RESULTS_DIR / "discrepancy.json", "w") as f:
        json.dump(discrepancy, f, indent=2, default=float)

    env = {"python": platform.python_version(), "platform": platform.platform(),
           "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
           "matplotlib": matplotlib.__version__, "dataset": str(DATA_PATH.relative_to(REPO_ROOT)),
           "dataset_sha256": file_sha256(DATA_PATH), "random_state": RANDOM_STATE,
           "extra_split_seeds": SEEDS, "command": "python -m src.run_baselines"}
    with open(RESULTS_DIR / "environment.json", "w") as f:
        json.dump(env, f, indent=2)

    log_path = RESULTS_DIR / "experiment_log.csv"
    new_log = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["timestamp", "command", "python", "sklearn", "alpha", "lambda", "n_iters",
                        "majority_valid_f1", "stratified_valid_f1", "softmax_valid_f1",
                        "softmax_valid_f1_mean_10", "softmax_valid_f1_std_10", "sklearn_agreement",
                        "with_2nd_sem_valid_f1", "seconds"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), "python -m src.run_baselines " + " ".join(sys.argv[1:]),
                    platform.python_version(),
                    sklearn.__version__, optimal_alpha, optimal_lambda, N_ITERS,
                    round(majority_row["valid_macro_f1"], 4), round(stratified_row["valid_macro_f1"], 4),
                    round(softmax_row["valid_macro_f1"], 4),
                    round(seed_summary.loc["Softmax regression (NumPy GD)", "mean"], 4),
                    round(seed_summary.loc["Softmax regression (NumPy GD)", "std"], 4),
                    round(agreement, 4), round(second_sem_row["valid_macro_f1"], 4),
                    round(time.perf_counter() - start, 1)])

    print(f"Done in {time.perf_counter() - start:.0f} s. Results in {RESULTS_DIR.relative_to(REPO_ROOT)}/")


def _markdown_table(df):
    """
    Parameters:
        `df` - a DataFrame
    Returns: df as a GitHub markdown table (so we don't need the tabulate package)
    """
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
