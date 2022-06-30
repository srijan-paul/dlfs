import sys
import numpy as np
from typing import Dict
import random
import math

def rows(mat: np.ndarray) -> int:
    return mat.shape[0]

def cols(mat: np.ndarray) -> int:
    return mat.shape[1]

def linear_regression_forward(X_batch: np.ndarray,
                              y_batch: np.ndarray,
                              weights: np.ndarray,
                              bias: float) -> float:
    assert rows(X_batch) == rows(
        y_batch), f"{rows(X_batch)} != {rows(y_batch)}"
    assert y_batch.shape[1] == 1, "target vector must have exactly 1 column"
    assert cols(X_batch) == rows(
        weights), f"{cols(X_batch)} != {rows(weights)}"

    unbiased_prediction = np.dot(X_batch, weights)
    prediction = unbiased_prediction + bias
    if (math.isnan(weights[0][0])): sys.exit(1)
    loss = np.mean(np.power((prediction - y_batch), 2))

    return loss, prediction, unbiased_prediction


def lg_train(X: np.ndarray, y: np.ndarray, learn_rate: float = 0.00001, batch_size: int = 10, iters: int = 2000):
    """Train the linear regression model on a dataset with samples `X` and target `y`"""

    assert X.ndim == 2 and y.ndim == 2, f"{X.ndim} {y.ndim}"
    assert y.shape[1] == 1, "target vector must have exactly 1 column"
    assert rows(X) == rows(y), "dimensions of samples and targets do not match"

    def init_weights() -> np.ndarray:
        """Initialize a random weight vector"""
        num_weights = cols(X)
        return np.random.randn(num_weights, 1)

    def loss_gradients(batch: Dict[str, np.ndarray], # X, y
                        prediction: np.ndarray, # P
                        unbiased_prediction: np.ndarray): # N
        X_, y_ = batch['X'], batch['y']

        grad_pred__unbiased_pred = np.ones_like(unbiased_prediction.shape[0])
        grad_pred__bias = np.ones_like((unbiased_prediction.shape[0], 1))

        # dPdW = dPdN * dNdW = [1, 1, 1..] * trans(X)
        grad_pred__weights = np.transpose(X_, (1, 0)) * grad_pred__unbiased_pred 
        # dLdP = 2 (P - y)
        grad_loss__pred = 2 * (prediction - y_)

        # dLdB = dLdP * dPdB
        grad_loss__bias = grad_loss__pred * grad_pred__bias

        # dLdW = dPdW * dLdP
        grad_loss__weights = np.dot(grad_pred__weights, grad_loss__pred)

        return grad_loss__weights, grad_loss__bias

    weights = init_weights()
    bias = random.randint(0, 10)
    for _ in range(iters):
        # In each iteration, we extract some random sample data and their corresponding
        # targets and group it as one 'batch'.
        batch: Dict[str, np.ndarray] = dict()
        start_index = random.randint(0, rows(X) - batch_size - 1)
        end_index = start_index + batch_size
        batch['X'] = X[start_index: end_index]
        batch['y'] = y[start_index: end_index]

        loss, pred, unbiased_pred = \
            linear_regression_forward(
                batch['X'],
                batch['y'],
                weights,
                bias
            )

        grad_w, grad_b = loss_gradients(batch, pred, unbiased_pred)
        weights -= learn_rate * grad_w
        bias -= learn_rate * grad_b[0][0]

    return weights, bias


def lg_predict(X, y, test_ratio: float = 0.3, iters: int = 1000):
    train_size = math.floor((1 - test_ratio) * X.shape[0])

    train_X = X[:train_size]
    train_y = y[:train_size]
    test_X = X[train_size:]
    test_y = y[train_size:]

    weights, bias = lg_train(train_X, train_y, 0.00001, 10, iters)
    pred = np.dot(test_X, weights) + bias
    return weights, bias, pred, test_y
