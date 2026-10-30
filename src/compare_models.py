"""
Milestone 4: compare kNN and a decision tree with the Milestone 3 baselines.

    python -m src.compare_models      # ~2 min, writes results/m4/

Same split, same preprocessing and same macro-F1 protocol as M3. Hyperparameters are
picked on the validation set only, and the test set is never scored here: it's used
once, in Milestone 5 (Week 3 p.51-52).
"""
import csv
import json
import platform
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import tree as treeViz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.data import CLASS_ORDER, RANDOM_STATE, REPO_ROOT, clean, load_raw, load_splits, undersample
from src.evaluate import evaluate, plot_confusion
from src.models import SoftmaxRegression
from src.run_baselines import row, run_naive

RESULTS_DIR = REPO_ROOT / "results"
M4_DIR = RESULTS_DIR / "m4"

# kNN search space: odd k so a two-way tie can't happen, up to sqrt(N) ~ 55 (W3 p.43),
# Euclidean distance from the lecture (W3 p.15) and cosine distance from Lab 3
K_LIST = list(range(1, 57, 2))
METRICS = ["euclidean", "cosine"]

# decision tree search space: the exact lists from Lab 4
CRITERIONS = ["entropy", "gini"]
MAX_DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30]
MIN_SAMPLES_SPLIT = [2, 4, 8, 16, 32, 64, 128, 256, 512]

SEEDS = list(range(10))

# readable columns for the error-analysis examples
SHOW_COLS = ["Course", "Age at enrollment", "Admission grade", "Tuition fees up to date",
             "Scholarship holder", "Debtor", "Curricular units 1st sem (enrolled)",
             "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)"]


def select_knn_model(X_train, X_valid, t_train, t_valid, k, metric):
    """
    Fit a k-NN classifier and return train and validation macro-F1 (the Lab 3 helper,
    scored with our primary metric instead of accuracy).

    Parameters:
        `X_train`, `t_train` - training data
        `X_valid`, `t_valid` - validation data
        `k` - number of neighbours
        `metric` - "euclidean" or "cosine"
    Returns: (train_f1, valid_f1)
    """
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(X_train, t_train)
    train_f1 = evaluate(t_train, knn.predict(X_train))["macro_f1"]
    valid_f1 = evaluate(t_valid, knn.predict(X_valid))["macro_f1"]
    return train_f1, valid_f1


def build_all_models(max_depths, min_samples_split, criterion, X_train, t_train, X_valid, t_valid):
    """
    Fit one decision tree per (max_depth, min_samples_split) pair (Lab 4).

    Parameters:
        `max_depths` - max_depth values to try
        `min_samples_split` - min_samples_split values to try
        `criterion` - "entropy" (information gain, W3 p.108) or "gini"
        `X_train`, `t_train` - training data
        `X_valid`, `t_valid` - validation data
    Returns: a dictionary `out` keyed by (max_depth, min_samples_split), where
             out[(d, s)]['val'] = validation macro-F1 and out[(d, s)]['train'] = training macro-F1
    """
    out = {}
    for d in max_depths:
        for s in min_samples_split:
            out[(d, s)] = {}
            tree = DecisionTreeClassifier(criterion=criterion, max_depth=d, min_samples_split=s,
                                          random_state=RANDOM_STATE)
            tree.fit(X_train, t_train)
            out[(d, s)]["val"] = evaluate(t_valid, tree.predict(X_valid))["macro_f1"]
            out[(d, s)]["train"] = evaluate(t_train, tree.predict(X_train))["macro_f1"]
    return out


def timed_fit(model, X, t):
    """
    Parameters:
        `model` - an unfitted model with .fit(X, t)
        `X`, `t` - training data
    Returns: (fitted model, fit time in seconds)
    """
    start = time.perf_counter()
    model.fit(X, t)
    return model, time.perf_counter() - start


def subgroup_table(models, X_valid, t_valid):
    """
    Preliminary subgroup check from the M2 plan (section 7.6), on the validation set.

    Parameters:
        `models` - dict of name -> fitted model
        `X_valid`, `t_valid` - validation data (binary / one-hot columns are still 0/1)
    Returns: a DataFrame with one row per (attribute, group, model): n, macro-F1, Dropout recall
    """
    groups = {
        "Gender": ("Gender", {0: "female", 1: "male"}),
        "Scholarship holder": ("Scholarship holder", {0: "no", 1: "yes"}),
        "Debtor": ("Debtor", {0: "no", 1: "yes"}),
        "Tuition fees up to date": ("Tuition fees up to date", {0: "no", 1: "yes"}),
        "Previous qualification": ("Previous qualification_1", {1: "secondary", 0: "other"}),
    }
    rows = []
    for attribute, (col, names) in groups.items():
        for value, name in names.items():
            mask = (X_valid[col] == value).to_numpy()
            for model_name, model in models.items():
                scores = evaluate(t_valid[mask], model.predict(X_valid[mask]))
                rows.append({"attribute": attribute, "group": name, "n": int(mask.sum()),
                             "model": model_name, "macro_f1": scores["macro_f1"],
                             "recall_Dropout": scores["recall_Dropout"],
                             "recall_Enrolled": scores["recall_Enrolled"]})
    return pd.DataFrame(rows)


def main():
    start = time.perf_counter()
    M4_DIR.mkdir(parents=True, exist_ok=True)

    s = load_splits(seed=RANDOM_STATE)
    X_train, X_valid, t_train, t_valid = s["X_train"], s["X_valid"], s["t_train"], s["t_valid"]
    print("train / valid / test:", len(X_train), len(X_valid), len(s["X_test"]), "(test not used)")

    # 1. baselines from M3: naive benchmarks + the tuned softmax regression
    with open(RESULTS_DIR / "discrepancy.json") as f:
        chosen = json.load(f)["chosen_hyperparameters"]
    majority_row, stratified_row = run_naive(s, RANDOM_STATE)
    softmax, softmax_time = timed_fit(
        SoftmaxRegression(alpha=chosen["alpha"], lam=chosen["lambda"], n_iters=chosen["n_iters"]),
        X_train, t_train)

    # 2. kNN: tune k and the distance on validation (Lab 3)
    knn_rows = []
    for metric in METRICS:
        for k in K_LIST:
            tr, va = select_knn_model(X_train, X_valid, t_train, t_valid, k, metric)
            knn_rows.append({"metric": metric, "k": k, "train_macro_f1": tr, "valid_macro_f1": va})
    knn_grid = pd.DataFrame(knn_rows)
    knn_grid.to_csv(M4_DIR / "knn_grid.csv", index=False)

    best_knn = knn_grid.loc[knn_grid["valid_macro_f1"].idxmax()]
    best_k, best_metric = int(best_knn["k"]), best_knn["metric"]
    print(f"Best kNN: k = {best_k}, metric = {best_metric}, valid macro-F1 = {best_knn['valid_macro_f1']:.4f}")
    knn, knn_time = timed_fit(KNeighborsClassifier(n_neighbors=best_k, metric=best_metric), X_train, t_train)

    # 3. decision tree: tune criterion, max_depth, min_samples_split on validation (Lab 4)
    best_params = {}   # criterion -> (max_depth, min_samples_split, train_f1, valid_f1)
    tree_rows = []
    for criterion in CRITERIONS:
        res = build_all_models(MAX_DEPTHS, MIN_SAMPLES_SPLIT, criterion, X_train, t_train, X_valid, t_valid)

        optimal_max_depth = 0
        optimal_min_samples_split = 0
        highest_validation_f1_so_far = 0
        for (d, sp) in res:
            tree_rows.append({"criterion": criterion, "max_depth": d, "min_samples_split": sp,
                              "train_macro_f1": res[(d, sp)]["train"], "valid_macro_f1": res[(d, sp)]["val"]})
            if res[(d, sp)]["val"] > highest_validation_f1_so_far:
                highest_validation_f1_so_far = res[(d, sp)]["val"]
                optimal_max_depth = d
                optimal_min_samples_split = sp
        best_params[criterion] = (optimal_max_depth, optimal_min_samples_split,
                                  res[(optimal_max_depth, optimal_min_samples_split)]["train"],
                                  highest_validation_f1_so_far)
        print(f"Criterion {criterion}: max_depth = {optimal_max_depth}, min_samples_split = "
              f"{optimal_min_samples_split}, valid macro-F1 = {highest_validation_f1_so_far:.4f}")
    tree_grid = pd.DataFrame(tree_rows)
    tree_grid.to_csv(M4_DIR / "tree_grid.csv", index=False)

    best_criterion = max(best_params, key=lambda c: best_params[c][3])
    best_d, best_s = best_params[best_criterion][:2]
    tree, tree_time = timed_fit(
        DecisionTreeClassifier(criterion=best_criterion, max_depth=best_d, min_samples_split=best_s,
                               random_state=RANDOM_STATE), X_train, t_train)
    print(f"Best tree: criterion = {best_criterion}, max_depth = {best_d}, min_samples_split = {best_s}, "
          f"actual depth = {tree.get_depth()}, leaves = {tree.get_n_leaves()}")

    # 4. model comparison table on the official split
    models = {"Softmax regression (M3 baseline)": softmax, "kNN": knn, "Decision tree": tree}
    times = {"Softmax regression (M3 baseline)": softmax_time, "kNN": knn_time, "Decision tree": tree_time}
    notes = {"Softmax regression (M3 baseline)": f"alpha={chosen['alpha']}, lambda={chosen['lambda']}, "
                                                 f"{chosen['n_iters']} iters",
             "kNN": f"k={best_k}, metric={best_metric}",
             "Decision tree": f"{best_criterion}, max_depth={best_d}, min_samples_split={best_s} "
                              f"(depth {tree.get_depth()}, {tree.get_n_leaves()} leaves)"}
    table_rows = [majority_row, stratified_row]
    for name, model in models.items():
        r = row(name, RANDOM_STATE, t_train, model.predict(X_train), t_valid, model.predict(X_valid), notes[name])
        r["fit_seconds"] = times[name]
        table_rows.append(r)
    table = pd.DataFrame(table_rows)

    # 5. variability: refit each tuned model on split seeds 0-9 (no re-tuning). The
    # softmax / naive seed runs are the M3 ones in results/seed_runs.csv
    seed_rows = []
    for seed in SEEDS:
        ss = load_splits(seed=seed)
        for name, model in [("kNN", KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)),
                            ("Decision tree", DecisionTreeClassifier(criterion=best_criterion, max_depth=best_d,
                                                                     min_samples_split=best_s,
                                                                     random_state=RANDOM_STATE))]:
            model.fit(ss["X_train"], ss["t_train"])
            seed_rows.append(row(name, seed, ss["t_train"], model.predict(ss["X_train"]),
                                 ss["t_valid"], model.predict(ss["X_valid"])))
    m3_seeds = pd.read_csv(RESULTS_DIR / "seed_runs.csv").replace(
        {"model": {"Softmax regression (NumPy GD)": "Softmax regression (M3 baseline)"}})
    seed_df = pd.concat([m3_seeds, pd.DataFrame(seed_rows)], ignore_index=True)
    seed_df.to_csv(M4_DIR / "seed_runs.csv", index=False)
    seed_summary = seed_df.groupby("model")["valid_macro_f1"].agg(["mean", "std", "min", "max"])
    # paired comparison: every model is scored on the same 10 splits, so compare them split by split
    per_seed = seed_df.pivot(index="split_seed", columns="model", values="valid_macro_f1")
    paired = []
    for a_name, b_name in [("Softmax regression (M3 baseline)", "Decision tree"),
                           ("Softmax regression (M3 baseline)", "kNN"), ("Decision tree", "kNN")]:
        diff = per_seed[a_name] - per_seed[b_name]
        paired.append({"model_a": a_name, "model_b": b_name, "a_wins_out_of_10": int((diff > 0).sum()),
                       "mean_diff": diff.mean(), "std_diff": diff.std(), "min_diff": diff.min(),
                       "max_diff": diff.max()})
    pd.DataFrame(paired).to_csv(M4_DIR / "paired_seed_comparison.csv", index=False)

    table["valid_macro_f1_mean_10_seeds"] = table["model"].map(seed_summary["mean"])
    table["valid_macro_f1_std_10_seeds"] = table["model"].map(seed_summary["std"])
    table.to_csv(M4_DIR / "model_comparison.csv", index=False)

    show = table[["model", "train_macro_f1", "valid_macro_f1", "valid_macro_f1_mean_10_seeds",
                  "valid_macro_f1_std_10_seeds", "valid_accuracy", "valid_recall_Dropout",
                  "valid_recall_Enrolled", "valid_recall_Graduate", "fit_seconds"]].round(3)
    show = show.astype(object).where(show.notna(), "-")
    with open(M4_DIR / "model_comparison.md", "w") as f:
        f.write("Validation results, official split (seed 42). Mean/std over split seeds 0-9. "
                "Test set not used.\n\n")
        lines = ["| " + " | ".join(show.columns) + " |", "|" + "---|" * len(show.columns)]
        lines += ["| " + " | ".join(str(v) for v in r.values) + " |" for _, r in show.iterrows()]
        f.write("\n".join(lines) + "\n")
    print(show.to_string(index=False))

    # 6. Enrolled: Lab 2 undersampling of the training set only, and class weights for the tree
    X_bal, t_bal = undersample(X_train, t_train, seed=RANDOM_STATE)
    variants = [
        ("Softmax regression (M3 baseline)", "none", softmax),
        ("Softmax regression (M3 baseline)", "undersampled train",
         SoftmaxRegression(alpha=chosen["alpha"], lam=chosen["lambda"], n_iters=chosen["n_iters"]).fit(X_bal, t_bal)),
        ("kNN", "none", knn),
        ("kNN", "undersampled train", KNeighborsClassifier(n_neighbors=best_k, metric=best_metric).fit(X_bal, t_bal)),
        ("Decision tree", "none", tree),
        ("Decision tree", "undersampled train",
         DecisionTreeClassifier(criterion=best_criterion, max_depth=best_d, min_samples_split=best_s,
                                random_state=RANDOM_STATE).fit(X_bal, t_bal)),
        ("Decision tree", "class_weight=balanced",
         DecisionTreeClassifier(criterion=best_criterion, max_depth=best_d, min_samples_split=best_s,
                                class_weight="balanced", random_state=RANDOM_STATE).fit(X_train, t_train)),
    ]
    imb_rows = []
    for name, fix, model in variants:
        va = evaluate(t_valid, model.predict(X_valid))
        imb_rows.append({"model": name, "imbalance_fix": fix,
                         "n_train": len(t_bal) if fix == "undersampled train" else len(t_train),
                         "valid_macro_f1": va["macro_f1"], "valid_accuracy": va["accuracy"],
                         "recall_Dropout": va["recall_Dropout"], "recall_Enrolled": va["recall_Enrolled"],
                         "recall_Graduate": va["recall_Graduate"], "f1_Enrolled": va["f1_Enrolled"]})
    imb = pd.DataFrame(imb_rows)
    imb.to_csv(M4_DIR / "imbalance.csv", index=False)
    print(imb.round(3).to_string(index=False))

    # 7. preliminary subgroup check
    sub = subgroup_table(models, X_valid, t_valid)
    sub.to_csv(M4_DIR / "subgroups.csv", index=False)

    # 8. error analysis: a few failures (Enrolled students we missed) and successes
    # (Dropouts we caught) from the best non-linear model, with readable raw values
    best_name = max(["kNN", "Decision tree"],
                    key=lambda n: table.loc[table["model"] == n, "valid_macro_f1"].item())
    y_best = pd.Series(models[best_name].predict(X_valid), index=X_valid.index)
    raw = clean(load_raw())
    missed_enrolled = y_best[(t_valid == 1) & (y_best != 1)].index[:4]
    caught_dropout = y_best[(t_valid == 0) & (y_best == 0)].index[:2]
    examples = raw.loc[list(missed_enrolled) + list(caught_dropout), SHOW_COLS + ["Target"]].copy()
    examples["Predicted"] = [CLASS_ORDER[i] for i in y_best.loc[examples.index]]
    examples["model"] = best_name
    examples.to_csv(M4_DIR / "error_examples.csv", index_label="row_id")

    # 9. figures
    # kNN train vs valid over k (Lab 3 curve), best metric
    g = knn_grid[knn_grid["metric"] == best_metric]
    plt.figure(figsize=(6, 3.5))
    plt.plot(g["k"], g["train_macro_f1"], label="Train")
    plt.plot(g["k"], g["valid_macro_f1"], label="Valid")
    plt.axhline(majority_row["valid_macro_f1"], color="grey", linestyle="--", label="Majority class (valid)")
    plt.title(f"kNN macro-F1 vs k ({best_metric} distance)")
    plt.xlabel("k"); plt.ylabel("Macro-F1"); plt.legend()
    plt.tight_layout(); plt.savefig(M4_DIR / "knn_curve.png", dpi=150); plt.close()

    # tree train vs valid over max_depth at the chosen min_samples_split (Lab 4 under/overfit)
    g = tree_grid[(tree_grid["criterion"] == best_criterion) & (tree_grid["min_samples_split"] == best_s)]
    plt.figure(figsize=(6, 3.5))
    plt.plot(g["max_depth"], g["train_macro_f1"], marker="o", label="Train")
    plt.plot(g["max_depth"], g["valid_macro_f1"], marker="o", label="Valid")
    plt.axhline(majority_row["valid_macro_f1"], color="grey", linestyle="--", label="Majority class (valid)")
    plt.title(f"Decision tree macro-F1 vs max_depth\n({best_criterion}, min_samples_split = {best_s})")
    plt.xlabel("max_depth"); plt.ylabel("Macro-F1"); plt.legend()
    plt.tight_layout(); plt.savefig(M4_DIR / "tree_curve.png", dpi=150); plt.close()

    # confusion matrices side by side
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, model) in zip(axes, models.items()):
        plot_confusion(t_valid, model.predict(X_valid), name, ax=ax)
    plt.tight_layout(); plt.savefig(M4_DIR / "confusion_all.png", dpi=150); plt.close()

    # top-10 tree feature importances
    imp = pd.Series(tree.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10)
    imp.to_csv(M4_DIR / "tree_feature_importance.csv", header=["importance"])
    plt.figure(figsize=(7, 4))
    plt.barh(imp.index[::-1], imp.values[::-1])
    plt.title("Decision tree: top 10 feature importances")
    plt.xlabel("Importance (total impurity decrease)")
    plt.tight_layout(); plt.savefig(M4_DIR / "tree_feature_importance.png", dpi=150); plt.close()

    # the first 3 levels of the tree (Lab 4 matplotlib fallback for visualize_tree)
    plt.figure(figsize=(24, 10))
    treeViz.plot_tree(tree, feature_names=list(X_train.columns), max_depth=3, class_names=CLASS_ORDER,
                      filled=True, rounded=True, fontsize=8)
    plt.savefig(M4_DIR / "tree_top3.png", dpi=110, bbox_inches="tight"); plt.close()

    # validation macro-F1 per model across seeds
    plt.figure(figsize=(7, 3.5))
    order = ["Majority class", "Stratified random", "Softmax regression (M3 baseline)", "kNN", "Decision tree"]
    for i, name in enumerate(order):
        vals = seed_df.loc[seed_df["model"] == name, "valid_macro_f1"]
        plt.scatter(vals, [i] * len(vals))
    plt.yticks(range(len(order)), order)
    plt.axvline(0.272, color="grey", linestyle="--", label="M2 success bar (0.272)")
    plt.xlabel("Validation macro-F1 (split seeds 0-9)"); plt.legend()
    plt.tight_layout(); plt.savefig(M4_DIR / "seed_spread.png", dpi=150); plt.close()

    # Enrolled recall with and without the imbalance fixes
    labels = [f"{m.replace(' regression (M3 baseline)', '')}: {f}" for m, f in zip(imb["model"], imb["imbalance_fix"])]
    y = np.arange(len(imb))[::-1]
    plt.figure(figsize=(7, 4))
    plt.barh(y + 0.2, imb["recall_Enrolled"], height=0.4, label="Enrolled recall")
    plt.barh(y - 0.2, imb["valid_macro_f1"], height=0.4, label="Macro-F1")
    plt.yticks(y, labels)
    plt.xlabel("Validation score"); plt.xlim(0, 1); plt.legend(loc="center right")
    plt.title("Effect of the imbalance fixes")
    plt.tight_layout(); plt.savefig(M4_DIR / "imbalance.png", dpi=150); plt.close()

    # 10. chosen settings + experiment log
    summary = {"knn": {"k": best_k, "metric": best_metric},
               "tree": {"criterion": best_criterion, "max_depth": int(best_d), "min_samples_split": int(best_s),
                        "actual_depth": int(tree.get_depth()), "leaves": int(tree.get_n_leaves())},
               "softmax": chosen, "best_nonlinear": best_name,
               "python": platform.python_version(), "scikit-learn": sklearn.__version__,
               "command": "python -m src.compare_models"}
    with open(M4_DIR / "chosen.json", "w") as f:
        json.dump(summary, f, indent=2)

    def valid_f1(name):
        return round(table.loc[table["model"] == name, "valid_macro_f1"].item(), 4)

    log_path = M4_DIR / "experiment_log.csv"
    new_log = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["timestamp", "command", "python", "sklearn", "knn_k", "knn_metric", "tree_criterion",
                        "tree_max_depth", "tree_min_samples_split", "softmax_valid_f1", "knn_valid_f1",
                        "tree_valid_f1", "knn_valid_f1_mean_10", "tree_valid_f1_mean_10", "seconds"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), "python -m src.compare_models",
                    platform.python_version(), sklearn.__version__, best_k, best_metric, best_criterion,
                    best_d, best_s, valid_f1("Softmax regression (M3 baseline)"), valid_f1("kNN"),
                    valid_f1("Decision tree"), round(seed_summary.loc["kNN", "mean"], 4),
                    round(seed_summary.loc["Decision tree", "mean"], 4), round(time.perf_counter() - start, 1)])

    print(f"Done in {time.perf_counter() - start:.0f} s. Results in results/m4/")


if __name__ == "__main__":
    main()
