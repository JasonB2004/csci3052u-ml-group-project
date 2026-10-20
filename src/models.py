"""
Course-level baseline: multi-class logistic (softmax) regression trained with
gradient descent, written from the Week 5 "Linear Models Part 2" slides.

    z = W x                         (W5 p.91, bias absorbed with a dummy x0 = 1, p.45)
    y_k = e^{z_k} / sum_k' e^{z_k'} (softmax, p.93)
    L_CE(y, t) = -sum_k t_k log y_k (cross-entropy with one-hot t, p.90 and p.94)
    dL_CE/dw_k = (y_k - t_k) x      (p.96)

The cost adds the Week 4 L2 regularizer (lambda/2)||W||^2 (W4 p.75), which turns
the update into weight decay (W4 p.88). Following the study-guide note, the bias
row is not penalized.
"""
import numpy as np


def softmax(z):
    """
    Parameters:
        `z` - logits, shape (N, K)
    Returns: probabilities, shape (N, K); each row sums to 1
    """
    # subtracting the row max doesn't change the answer (it cancels in the ratio)
    # but stops e^z from overflowing for large logits
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def one_hot(t, K):
    """
    Parameters:
        `t` - integer labels in {0, ..., K-1}, shape (N,)
        `K` - number of classes
    Returns: one-of-K encoding of t, shape (N, K) (W5 p.90)
    """
    T = np.zeros((len(t), K))
    T[np.arange(len(t)), t] = 1
    return T


def add_bias_column(X):
    """
    Add the dummy feature x0 = 1 so the bias becomes w0 (W4 p.32, W5 p.45).

    Parameters:
        `X` - design matrix, shape (N, D)
    Returns: shape (N, D+1) with a column of 1's in front
    """
    return np.hstack([np.ones((X.shape[0], 1)), X])


class SoftmaxRegression:
    """
    Softmax regression with L2 regularization, fit by full-batch gradient descent.

    Parameters:
        `alpha` - learning rate (W4 p.84: typically 0.001 to 0.1, grid search it)
        `lam` - L2 regularization strength lambda (W4 p.76: a hyperparameter we tune on validation)
        `n_iters` - number of gradient descent steps (W5 p.26: stop after a # of iterations)
    """

    def __init__(self, alpha=0.1, lam=0.0, n_iters=2000):
        self.alpha = alpha
        self.lam = lam
        self.n_iters = n_iters

    def cost(self, X1, T):
        """
        Parameters:
            `X1` - design matrix with the bias column, shape (N, D+1)
            `T` - one-hot targets, shape (N, K)
        Returns: J(W) = average cross-entropy + (lambda/2) * sum of squared weights (no bias row)
        """
        Y = softmax(X1 @ self.W)
        ce = -np.sum(T * np.log(Y + 1e-12)) / X1.shape[0]
        return ce + (self.lam / 2) * np.sum(self.W[1:] ** 2)

    def gradient(self, X1, T):
        """
        Parameters:
            `X1` - design matrix with the bias column, shape (N, D+1)
            `T` - one-hot targets, shape (N, K)
        Returns: dJ/dW, shape (D+1, K). The data part is (1/N) X^T (Y - T), which is
                 the (y_k - t_k) x rule from W5 p.96 for every example at once
        """
        Y = softmax(X1 @ self.W)
        grad = X1.T @ (Y - T) / X1.shape[0]
        grad[1:] += self.lam * self.W[1:]   # regularizer gradient, bias row left alone
        return grad

    def fit(self, X, t):
        """
        Parameters:
            `X` - training design matrix, shape (N, D) (already normalized)
            `t` - integer labels, shape (N,)
        Returns: self, with self.W of shape (D+1, K) and self.cost_history (one J per step)
        """
        X1 = add_bias_column(np.asarray(X, dtype=float))
        t = np.asarray(t)
        self.K = int(t.max()) + 1
        T = one_hot(t, self.K)

        # W5 p.79: a standard initialization is w = 0. The CE loss is convex in W, so
        # with a small enough alpha GD converges to the same optimum from any start
        self.W = np.zeros((X1.shape[1], self.K))
        self.cost_history = [self.cost(X1, T)]
        for _ in range(self.n_iters):
            self.W = self.W - self.alpha * self.gradient(X1, T)
            self.cost_history.append(self.cost(X1, T))
        return self

    def predict_proba(self, X):
        """
        Parameters:
            `X` - design matrix, shape (N, D)
        Returns: class probabilities y, shape (N, K)
        """
        return softmax(add_bias_column(np.asarray(X, dtype=float)) @ self.W)

    def predict(self, X):
        """
        Parameters:
            `X` - design matrix, shape (N, D)
        Returns: predicted labels, the argmax of z (W5 p.92), shape (N,)
        """
        return np.argmax(self.predict_proba(X), axis=1)
