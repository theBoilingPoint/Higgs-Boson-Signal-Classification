import numpy as np
from implementations import sigmoid
from proj1_helpers import format_result


def split_by_k(y, x, k_indices, k):
    """Split data y, x for k-folding

    Args:
        y (np.array): (N, ) prediction vector
        x (np.array): (N, D) input data
        k_indices (np.array): indices for k-folding
        k (int): index to be considered as test

    Returns:
        np.array, np.array, np.array, np.array: splitted data
    """
    mask_test = np.zeros_like(y, bool)
    mask_test[k_indices[k]] = True
    test_x = x[mask_test, :]
    test_y = y[mask_test]

    mask_train = np.ones_like(y, bool)
    mask_train[k_indices[k]] = False
    train_x = x[mask_train, :]
    train_y = y[mask_train]

    return train_x, train_y, test_x, test_y


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y (np.array): (N, ) prediction vector
        k_fold (int): number of k to fold the data.
        seed (int): random seed for reproducibility

    Returns:
        np.array: indices for k-fold
    """
    """"""

    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def make_prediction(tx, w, threshold=0, log=False, test=False):
    """Make prediction for model w

    Args:
        tx (np.array): (N, D) input data
        w (np.array): (D, ) weight of the model
        threshold (int, optional): threshold for prediction. Defaults to 0.
        log (bool, optional): Flag for doing sigmoid on inner product of tx and w for preprocessing data for logistic regression and regularized logistic regression. Defaults to False.
        test (bool, optional): Flag for making prediction for submission (y should be {-1, 1} regardless of the model). Defaults to False.

    Returns:
        [type]: [description]
    """
    pred = np.dot(tx, w)
    if log:
        pred = sigmoid(tx, w)
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0 if (log and not test) else -1

    return pred


def compute_acc(y, tx, w, threshold=0, log=False):
    """Compute accuracy for the model w

    Args:
        y (np.array): (N, ) prediction vector
        tx (np.array): (N, D) input data
        w (np.array): (D, ) weight of the model
        threshold (int, optional): threshold for prediction. Defaults to 0.
        log (bool, optional): Flag for doing sigmoid on inner product of tx and w for preprocessing data for logistic regression and regularized logistic regression. Defaults to False.

    Returns:
        [type]: [description]
    """
    pred = make_prediction(tx, w, threshold, log)

    corrects = y == pred
    corrects.astype(int)
    return np.sum(corrects) / y.shape[0]


def cross_validation(y, x, k_indices, k, trainer, compute_loss, threshold=0, log=False):
    """Run cross validaion for kth of k-fold

    Args:
        y (np.array): (N, ) prediction vector
        x (np.array): (N, D) input data
        k_indices (np.array): indices for k-folding
        k (int): index to be considered as test
        trainer (function): function that train a model and return weight and loss (y: np.array, x:np.array) => weight, loss. User will call any implementation with their hyperparameters e.g. trainer = lambda y, x: ridge_regression(y, x, lambda_=0.001)
        compute_loss (function): function that compute loss with charateristic of (y, x, weight) => loss.
        threshold (int, optional): threshold for prediction. Defaults to 0.
        log (bool, optional): Flag for doing sigmoid on inner product of tx and w for preprocessing data for logistic regression and regularized logistic regression. Defaults to False.

    Returns:
        [type]: [description]
    """
    train_x, train_y, test_x, test_y = split_by_k(y, x, k_indices, k)

    weight, loss_tr = trainer(train_y, train_x)

    loss_te = compute_loss(test_y, test_x, weight)

    acc_tr = compute_acc(train_y, train_x, weight, threshold, log)
    acc_te = compute_acc(test_y, test_x, weight, threshold, log)

    return acc_tr, acc_te, loss_tr, loss_te, weight


def run_k_fold(
    y, tx_train, trainer, compute_loss, threshold=0, log=False, k_fold=5, seed=1
):
    """Run k-fold testing and print a summary report

    Args:
        y (np.array): (N, ) prediction vector
        tx_train (np.array): (N, D) input data
        trainer (function): function that train a model and return weight and loss (y: np.array, x:np.array) => weight, loss. User will call any implementation with their hyperparameters e.g. trainer = lambda y, x: ridge_regression(y, x, lambda_=0.001)
        compute_loss (function): function that compute loss with charateristic of (y, x, weight) => loss.
        threshold (int, optional): threshold for prediction. Defaults to 0.
        log (bool, optional): Flag for doing sigmoid on inner product of tx and w for preprocessing data for logistic regression and regularized logistic regression. Defaults to False.
        k_fold (int, optional): number of k for fold and run. Defaults to 5.
        seed (int, optional): random seed for reproducibility. Defaults to 1.
    """
    loss_trs = []
    loss_tes = []
    acc_trs = []
    acc_tes = []
    ws = []

    k_indices = build_k_indices(y, k_fold, seed)
    for k in range(k_fold):
        acc_tr, acc_te, loss_tr, loss_te, weight = cross_validation(
            y,
            tx_train,
            k_indices,
            k,
            trainer,
            compute_loss,
            threshold=threshold,
            log=log,
        )

        loss_trs.append(loss_tr)
        loss_tes.append(loss_te)
        acc_trs.append(acc_tr)
        acc_tes.append(acc_te)
        ws.append(weight)

    mean_loss_tr = np.mean(loss_trs)
    mean_loss_te = np.mean(loss_tes)
    mean_acc_tr = np.mean(acc_trs)
    mean_acc_te = np.mean(acc_tes)

    var_loss_tr = np.var(loss_trs)
    var_loss_te = np.var(loss_tes)
    var_acc_tr = np.var(acc_trs)
    var_acc_te = np.var(acc_tes)

    print(
        format_result(
            mean_loss_tr,
            mean_loss_te,
            mean_acc_tr,
            mean_acc_te,
            var_acc_tr,
            var_acc_te,
            var_loss_tr,
            var_loss_te,
            for_spreadsheet=False,
        )
    )
