import numpy as np

from common import standardize, add_bias
from regression import load_data, mse_loss, gradient

BATCH_SIZES = (8, 32, 64)


def minibatch_gradient_descent(X, y, alpha, batch_size, n_iter=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    w = np.zeros(X.shape[1])
    loss_history = []
    for _ in range(n_iter):
        loss_history.append(mse_loss(X, y, w))  # full-dataset loss, so runs are comparable
        idx = rng.choice(n, size=batch_size, replace=False)
        w -= alpha * gradient(X[idx], y[idx], w)
    return w, loss_history


if __name__ == "__main__":
    X_raw, y = load_data("insurance_charges.csv")
    X_scaled, _, _ = standardize(X_raw)
    X = add_bias(X_scaled)

    for batch_size in BATCH_SIZES:
        w, loss_history = minibatch_gradient_descent(X, y, alpha=0.01, batch_size=batch_size)
        print(f"batch_size={batch_size}")
        print(f"  loss[0]  = {loss_history[0]:.4f}")
        print(f"  loss[-1] = {loss_history[-1]:.4f}")
        print(f"  weights  = {w}")
        print()

    print(f"full batch ({len(y)} samples) for comparison")
    w, loss_history = minibatch_gradient_descent(X, y, alpha=0.01, batch_size=len(y))
    print(f"  loss[0]  = {loss_history[0]:.4f}")
    print(f"  loss[-1] = {loss_history[-1]:.4f}")
