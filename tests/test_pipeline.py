"""
Sanity checks for the data pipeline and the softmax regression baseline.

    python -m tests.test_pipeline

Plain asserts, so no extra test package is needed. Every check prints a line when it passes.
"""
import numpy as np

from src.data import (DATA_PATH, DATA_SHA256, continuous_cols, file_sha256, load_splits)
from src.evaluate import evaluate
from src.models import SoftmaxRegression, add_bias_column, one_hot, softmax


def test_dataset_version():
    assert file_sha256(DATA_PATH) == DATA_SHA256
    print("ok  dataset SHA-256 matches the M2 version")


def test_split_matches_m2():
    s = load_splits()
    assert s["X"].shape == (4424, 169)
    assert (len(s["X_train"]), len(s["X_valid"]), len(s["X_test"])) == (3096, 664, 664)
    assert set(s["X_train"].index).isdisjoint(s["X_valid"].index)
    assert set(s["X_train"].index).isdisjoint(s["X_test"].index)
    assert set(s["X_valid"].index).isdisjoint(s["X_test"].index)
    assert not any("2nd sem" in c for c in s["X_train"].columns)

    # class balance from the M2 split table (Dropout / Enrolled / Graduate)
    for ts in [s["t_train"], s["t_valid"], s["t_test"]]:
        p = ts.value_counts(normalize=True).sort_index().round(2).tolist()
        assert p == [0.32, 0.18, 0.5], p
    print("ok  design matrix 169 columns, split 3096/664/664, disjoint, stratified, no 2nd-sem columns")


def test_normalization_uses_training_stats():
    s = load_splits()
    train = s["X_train"][continuous_cols]
    assert np.allclose(train.mean(), 0, atol=1e-6)
    assert np.allclose(train.std(ddof=0), 1, atol=1e-3)   # slightly under 1 because of epsilon
    # one-hot / binary columns are untouched
    other = s["X_train"].drop(columns=continuous_cols)
    assert set(np.unique(other.to_numpy())) <= {0, 1}
    print("ok  continuous columns have train mean 0 / std 1, binary columns still 0/1")


def test_softmax_probabilities():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(5, 3)) * 50   # big logits would overflow without the max trick
    y = softmax(z)
    assert np.all(np.isfinite(y))
    assert np.allclose(y.sum(axis=1), 1)
    print("ok  softmax rows sum to 1 and stay finite for large logits")


def test_gradient_check():
    # compare the analytic gradient (1/N) X^T (Y - T) + lambda W (W5 p.96) with a
    # finite-difference estimate of dJ/dW on a small random problem
    rng = np.random.default_rng(1)
    X1 = add_bias_column(rng.normal(size=(20, 4)))
    T = one_hot(rng.integers(0, 3, size=20), 3)
    model = SoftmaxRegression(lam=0.1)
    model.W = rng.normal(size=(5, 3))

    analytic = model.gradient(X1, T)
    numeric = np.zeros_like(model.W)
    h = 1e-6
    for i in range(model.W.shape[0]):
        for k in range(model.W.shape[1]):
            W0 = model.W.copy()
            model.W[i, k] = W0[i, k] + h
            J_plus = model.cost(X1, T)
            model.W[i, k] = W0[i, k] - h
            J_minus = model.cost(X1, T)
            model.W = W0
            numeric[i, k] = (J_plus - J_minus) / (2 * h)
    assert np.max(np.abs(analytic - numeric)) < 1e-6, np.max(np.abs(analytic - numeric))
    print("ok  analytic gradient matches finite differences (max diff < 1e-6)")


def test_cost_decreases():
    s = load_splits()
    model = SoftmaxRegression(alpha=0.3, lam=0.01, n_iters=200).fit(s["X_train"], s["t_train"])
    J = np.array(model.cost_history)
    assert np.all(np.diff(J) <= 1e-12)   # W4 p.83: GD always decreases J for a small enough alpha
    print(f"ok  training cost decreases every step ({J[0]:.3f} -> {J[-1]:.3f})")


def test_deterministic():
    scores = []
    for _ in range(2):
        s = load_splits()
        model = SoftmaxRegression(alpha=0.3, lam=0.01, n_iters=200).fit(s["X_train"], s["t_train"])
        scores.append(evaluate(s["t_valid"], model.predict(s["X_valid"]))["macro_f1"])
    assert scores[0] == scores[1]
    print(f"ok  two runs give the same validation macro-F1 ({scores[0]:.4f})")


if __name__ == "__main__":
    test_dataset_version()
    test_split_matches_m2()
    test_normalization_uses_training_stats()
    test_softmax_probabilities()
    test_gradient_check()
    test_cost_decreases()
    test_deterministic()
    print("All checks passed.")
