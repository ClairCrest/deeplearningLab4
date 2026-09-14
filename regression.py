import numpy as np
import pandas as pd

from common import standardize, add_bias

PREDICT_ROWS = np.array([
    [31, 0, 22],
    [43, 1, 19],
    [28, 1, 33],
    [50, 0, 28],
], dtype=float)


def load_data(path):
    df = pd.read_csv(path)
    X = df[["age", "gender", "bmi"]].to_numpy(dtype=float)
    y = df["charges"].to_numpy(dtype=float)
    return X, y


def predict(X, w):
    return X @ w


def mse_loss(X, y, w):
    err = predict(X, w) - y
    return np.mean(err ** 2)


def gradient(X, y, w):
    err = predict(X, w) - y
    return (2 / len(y)) * (X.T @ err)


def gradient_descent(X, y, alpha, n_iter=1000):
    w = np.zeros(X.shape[1])
    loss_history = []
    for _ in range(n_iter):
        loss_history.append(mse_loss(X, y, w))
        w -= alpha * gradient(X, y, w)
    return w, loss_history


if __name__ == "__main__":
    X_raw, y = load_data("insurance_charges.csv")
    X_scaled, mean, std = standardize(X_raw)
    X = add_bias(X_scaled)

    for alpha in (0.01, 0.1):
        w, loss_history = gradient_descent(X, y, alpha)
        print(f"alpha={alpha}")
        print(f"  loss[0]   = {loss_history[0]:.4f}")
        print(f"  loss[100] = {loss_history[100]:.4f}")
        print(f"  loss[-1]  = {loss_history[-1]:.4f}")
        print(f"  weights   = {w}")
        print()

    # required run: w0 = 0, alpha = 0.01
    w, _ = gradient_descent(X, y, alpha=0.01)
    rows_scaled, _, _ = standardize(PREDICT_ROWS, mean, std)
    rows_X = add_bias(rows_scaled)
    preds = predict(rows_X, w)

    print("predictions (alpha=0.01 weights):")
    for (age, gender, bmi), charge in zip(PREDICT_ROWS, preds):
        print(f"  age={age:.0f} gender={gender:.0f} bmi={bmi:.0f} -> charges={charge:.2f}")
