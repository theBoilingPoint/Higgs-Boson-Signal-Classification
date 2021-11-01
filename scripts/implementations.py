import numpy as np
from proj1_helpers import *


def compute_mse(y, tx, w):
    """Compute the loss by mse.

    Args:
        y (np.array): (N,) ground truth vector of N data
        tx (np.array): (N,D) data matrix of N data and D features
        w (np.array): (D,) weight vector for D features

    Returns:
        float: MSE Loss
    """
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse


# --- Linear Regression ---#


def compute_mse_gradient(y, tx, w):
    """Compute the gradient when loss function is mse.

    Args:
        y (np.array): (N,) ground truth vector of N data
        tx (np.array): (N,D) data matrix of N data and D features
        w (np.array): (D,) weight vector for D features

    Returns:
        float: gradient value
    """
    
    return -tx.T @ (y - tx @ w) / len(y)


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """function of least squares regression based on gradient descent.

    Args:
        y (np.array): (N,) ground truth vector of N data
        tx (np.array): (N,D) data matrix of N data and D features
        initial_w (np.array): initial weight vector of D features
        max_iters (int): maximum iteration time for gradient descent 
        gamma (float): iterative step size 

    Returns:
        float, float: last weight and last loss 
    """
    w = initial_w

    for i in range(max_iters):
        gradient = compute_mse_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        w -= gamma * gradient

    return w, loss


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """function of least squares regression based on stochastic gradient descent.

    Args:
        y (np.array): (N,) ground truth vector of N data
        tx (np.array): (N,D) data matrix of N data and D features
        initial_w (np.array): initial weight vector of D features
        max_iters (int): maximum iteration time for gradient descent 
        gamma (float): iterative step size 

    Returns:
        float, float: last weight and last loss 
    """
    w = initial_w
    batch_size = 1

    for i in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=batch_size):
            gradient = compute_mse_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_mse(y, tx, w)
            w -= gamma * gradient

    return w, loss


# --- Least Squares ---#


def least_squares(y, tx):
    """Compute closed-form of least square of y and tx

    Args:
        y (np.array): (N,) ground truth vector of N data
        tx (np.array): (N,D) data matrix of N data and D features

    Returns:
        tuple(np.array, float): (D,) weight and its corresponding loss
    """
    b = np.dot(tx.T, y)
    a = np.dot(tx.T, tx)
    w = np.linalg.solve(a, b)

    loss = compute_mse(y, tx, w)
    return w, loss


# --- Ridge Regression ---#


def ridge_regression(y, tx, lambda_):
    """Compute closed-form of ridge regression of y, tx and lambda
    Use MSE as loss
    Use L2-norm as regularizer

    Args:
        y (np.array): (N,) ground truth vector of N data
        tx (np.array): (N,D) data matrix of N data and D features
        lambda_ (float): regularization parameter

    Returns:
        tuple(np.array, float): (D,) weight and its corresponding loss
    """
    N, D = tx.shape
    b = np.dot(tx.T, y)  # NxD, N
    a = np.dot(tx.T, tx) + 2 * N * lambda_ * np.identity(D)  # DxN, NxD
    w = np.linalg.solve(a, b)

    mse_loss = compute_mse(y, tx, w)
    loss = mse_loss + lambda_ * np.linalg.norm(w) ** 2
    return w, loss


# --- Logistic Regression ---#


def _sigmoid(z):
    """Compute Sigmoid of floating point z

    Args:
        z (float): a number

    Returns:
        float: result of sigmoid
    """
    if z <= 0:
        return np.exp(z) / (1 + np.exp(z))
    result = 1 / (1 + np.exp(-z))
    return result


def sigmoid(tx, w):
    """Compute element-wise sigmoid of vector of inner product of tx and w

    Args:
        tx (np.array): (N,D) data matrix of N data and D features
        w (np.array): (D,) weight vector for D features

    Returns:
        np.array: vector of computed sigmoid
    """
    z = np.dot(tx, w)
    return np.vectorize(_sigmoid)(z)


def compute_grad_logistic_regression(y, tx, w):
    """Compute gradient for logistic regression

    Args:
        y (np.array): (N,) ground truth vector of N data
        tx (np.array): (N,D) data matrix of N data and D features
        w (np.array): (D,) weight vector for D features

    Returns:
        np.array: (D,) vector of computed gradient
    """
    grad = np.dot(tx.T, sigmoid(tx, w) - y)
    return grad


def compute_loss_logistic_regression(y, tx, w):
    """Compute loss for logistic regression

    Args:
        y (np.array): (N,) ground truth vector of N data
        tx (np.array): (N,D) data matrix of N data and D features
        w (np.array): (D,) weight vector for D features

    Returns:
        float: computed loss
    """
    xTw = np.array(np.dot(tx, w), dtype=np.longdouble)
    l = np.log(1 + np.exp(xTw)) - y * xTw
    return np.sum(l)


def loss_reg_logistic_regression(lambda_, w):
    """Compute regularizer term for logistic regression (lambda * L2-norm of weight)

    Args:
        lambda_ (float): regularization parameter
        w (np.array): (D,) weight vector for D features

    Returns:
        float: computed regularizer term
    """
    return lambda_ / 2 * np.linalg.norm(w) ** 2 / w.shape[0]


def grad_reg_logistic_regression(lambda_, w):
    """Compute regularizer term of logistic regression for gradient (lamda * weight)

    Args:
        lambda_ (float): lambda of how much complexity of the model should be penalized
        w (np.array): (D,) weight vector for D features

    Returns:
        np.array: (D,) vector of computed regularizer for gradient
    """
    return lambda_ * w


def iter_logistic_regression(y, tx, lambda_, w, gamma):
    """Compute 1 iteration of logistic regression
    Note: when lambda_ > 0, this will function as regularized logistic regression

    Args:
        y (np.array): (N,) ground truth vector of N data
        tx (np.array): (N,D) data matrix of N data and D features
        lambda_ (float): regularization parameter
        w (np.array): (D,) weight vector for D features
        gamma (float): learning rate for 1 iteration

    Returns:
        tuple(np.array, float): (D,) weight and its corresponding loss
    """
    loss = compute_loss_logistic_regression(y, tx, w) + loss_reg_logistic_regression(
        lambda_, w
    )
    grad = compute_grad_logistic_regression(y, tx, w) + grad_reg_logistic_regression(
        lambda_, w
    )
    w = w - gamma * grad
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma, verbose=0):
    """Compute logistic regression for `max_iters` iterations

    Args:
        y (np.array): (N,) ground truth vector of N data
        tx (np.array): (N,D) data matrix of N data and D features
        initial_w (np.array): (D,) initial weight vector for computing logistic regression
        max_iters (int): number of steps to run
        gamma (float): step size
        verbose (int, optional): Verbosity of outputing loss(0: output nothing, 1: every 100 iterations, 2: every iteration). Defaults to 0.

    Returns:
        tuple(np.array, float): last (D,) weight and its corresponding loss
    """
    return reg_logistic_regression(
        y, tx, 0, initial_w, max_iters, gamma, verbose=verbose
    )


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, verbose=0):
    """Compute logistic regression for `max_iters` iterations

    Args:
        y (np.array): (N,) ground truth vector of N data
        tx (np.array): (N,D) data matrix of N data and D features
        lambda_ (float): regularization parameter
        initial_w (np.array): (D,) initial weight vector for computing logistic regression
        max_iters (int): number of steps to run
        gamma (float): step size
        verbose (int, optional): Verbosity of outputing loss(0: output nothing, 1: every 100 iterations, 2: every iteration). Defaults to 0.

    Returns:
        tuple(np.array, float): last (D,) weight and its corresponding loss
    """
    w = initial_w
    loss = None

    for n_iter in range(max_iters):
        w, loss = iter_logistic_regression(y, tx, lambda_, w, gamma)
        if verbose == 2 or (verbose == 1 and n_iter % 100 == 0):
            print(
                "Gradient Descent({bi}/{ti}): loss={l}, gamma={gamma}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss, gamma=gamma
                )
            )

    return w, loss
