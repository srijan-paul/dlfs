from ast import Tuple
import sys
import numpy as np
from typing import Dict
import random
import math

data = np.loadtxt(open("./data.csv", "rb"), delimiter=",", skiprows=1)

X = np.delete(data, 3, 1)
y = data[:, 3] + 10  # y = 5*x1 + 3*x2 + x3 + 10 ( c = 10)
y = y.reshape((y.shape[0], 1))


def rows(mat: np.ndarray) -> int:
    return mat.shape[0]


def cols(mat: np.ndarray) -> int:
    return mat.shape[1]


def forward_linear_regression(X_batch: np.ndarray,
                              y_batch: np.ndarray,
                              weights: Dict[str, np.ndarray]
                              ) -> Tuple[float, Dict[str, np.ndarray]]:
    '''
    Forward pass for the step-by-step linear regression.
    '''
    # assert batch sizes of X and y are equal
    assert X_batch.shape[0] == y_batch.shape[0]

    # assert that matrix multiplication can work
    assert X_batch.shape[1] == weights['W'].shape[0]

    # assert that B is simply a 1x1 ndarray
    assert weights['B'].shape[0] == weights['B'].shape[1] == 1

    # compute the operations on the forward pass
    N = np.dot(X_batch, weights['W'])
    P = N + weights['B']

    loss = np.mean(np.power(y_batch - P, 2))

    # save the information computed on the forward pass
    forward_info: Dict[str, np.ndarray] = {}
    forward_info['X'] = X_batch
    forward_info['N'] = N
    forward_info['P'] = P
    forward_info['y'] = y_batch

    return loss, forward_info

def loss_gradients(forward_info: Dict[str, np.ndarray],
                   weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    '''
    Compute dLdW and dLdB for the step-by-step linear regression model.
    '''
    dLdP = -2 * (forward_info['y'] - forward_info['P'])
    dPdN = np.ones_like(forward_info['N'])
    dPdB = np.ones_like(weights['B'])
    dLdN = dLdP * dPdN
    dNdW = np.transpose(forward_info['X'], (1, 0))
    
    # need to use matrix multiplication here,
    # with dNdW on the left (see note at the end of last chapter)    
    dLdW = np.dot(dNdW, dLdN)

    # need to sum along dimension representing the batch size:
    # see note near the end of the chapter    
    dLdB = (dLdP * dPdB).sum(axis=0)

    loss_gradients: Dict[str, np.ndarray] = {}
    loss_gradients['W'] = dLdW
    loss_gradients['B'] = dLdB

    return loss_gradients

def lg_train(X, y, learn_rate: float = 0.01, iters: int = 1000):
  def init_weights() -> Dict[str, np.ndarray]:
    weights = dict()
    weights['W'] = np.random.randn(X.shape[1], 1)
    weights['B'] = np.random.randn(1, 1)
    return weights

  weights = init_weights()
  for _ in range(iters):
    pass

def lg_predict(train_X, train_y, test_X, test_y):
    weights, bias = lg_train(train_X, train_y)
    pred = np.dot(test_X, weights) + bias
    for i in range(test_y.shape[0]):
        print(pred[i] + bias, test_y[i])


train_size = math.floor(0.8 * X.shape[0])
test_size = X.shape[0] - train_size

lg_predict(X[:train_size],
           y[:train_size],
           X[train_size:],
           y[train_size:])
