import numpy as np
import pandas as pd

from common import standardize, add_bias

FEATURES = ["sepal length", "sepal width", "petal length", "petal width"]


def load_data(path):
    df = pd.read_csv(path)
    X = df[FEATURES].to_numpy(dtype=float)
    y = (df["class"] == "Iris-versicolor").astype(float).to_numpy()
    return X, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict_proba(X, w):
    return sigmoid(X @ w)


def bce_loss(X, y, w):
    p = predict_proba(X, w)
    eps = 1e-12
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


def gradient(X, y, w):
    p = predict_proba(X, w)
    return (X.T @ (p - y)) / len(y)


def gradient_descent(X, y, alpha, n_iter=1000):
    w = np.zeros(X.shape[1])
    loss_history = []
    for _ in range(n_iter):
        loss_history.append(bce_loss(X, y, w))
        w -= alpha * gradient(X, y, w)
    return w, loss_history


def confusion_matrix(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


if __name__ == "__main__":
    X_train_raw, y_train = load_data("iris_2class_train.csv")
    X_test_raw, y_test = load_data("iris_2class_test.csv")

    X_train_scaled, mean, std = standardize(X_train_raw)
    X_test_scaled, _, _ = standardize(X_test_raw, mean, std)

    X_train = add_bias(X_train_scaled)
    X_test = add_bias(X_test_scaled)

    w, loss_history = gradient_descent(X_train, y_train, alpha=0.1)
    print(f"loss[0]   = {loss_history[0]:.4f}")
    print(f"loss[100] = {loss_history[100]:.4f}")
    print(f"loss[-1]  = {loss_history[-1]:.4f}")
    print(f"weights   = {w}")

    y_pred = (predict_proba(X_test, w) >= 0.5).astype(float)
    tp, tn, fp, fn = confusion_matrix(y_test, y_pred)
    accuracy = (tp + tn) / len(y_test)

    print()
    print("confusion matrix, positive class = Iris-versicolor")
    print(f"  actual setosa:     predicted setosa={tn}  predicted versicolor={fp}")
    print(f"  actual versicolor: predicted setosa={fn}  predicted versicolor={tp}")
    print(f"accuracy = {accuracy:.4f}")
