import numpy as np

from common import standardize, add_bias


def test():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    Xs, mean, std = standardize(X)
    assert abs(Xs.mean()) < 1e-9

    Xb = add_bias(X)
    assert Xb.shape[1] == 2
    assert (Xb[:, 0] == 1).all()

    from regression import mse_loss, gradient
    Xr = add_bias(Xs)
    y = np.array([2.0, 4.0, 6.0, 8.0])
    w = np.zeros(Xr.shape[1])
    loss_before = mse_loss(Xr, y, w)
    w_after = w - 0.5 * gradient(Xr, y, w)
    loss_after = mse_loss(Xr, y, w_after)
    assert loss_after < loss_before

    from classification import sigmoid, confusion_matrix
    assert abs(sigmoid(0.0) - 0.5) < 1e-9
    y_true = np.array([1, 0, 1, 0], dtype=float)
    y_pred = np.array([1, 0, 0, 0], dtype=float)
    assert confusion_matrix(y_true, y_pred) == (1, 2, 0, 1)

    print("all checks passed")


if __name__ == "__main__":
    test()
