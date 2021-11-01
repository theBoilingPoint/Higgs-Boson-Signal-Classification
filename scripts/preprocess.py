import numpy as np
from proj1_helpers import *


def one_hot_encode(col):
    """Calculate one hot encoding for the input column vector

    Args:
        col (np.array): (N, ) column vector from input data

    Returns:
        np.array: (N, X) matrix of one hot encoding for the input column vector
    """
    assert len(col.shape) == 1
    n_classes = int(col.max() - col.min() + 1)

    return np.eye(n_classes)[col.astype(int)]


def normalize(tx, train, MEAN, STD):
    """Normalize input data

    Args:
        tx (np.array): (N, D) input data
        train (bool): True to compute new value; False to use the specified value
        MEAN (np.array): (1, D) mean of data
        STD (np.array): (1, D) std of data

    Returns:
        np.array: (N, D) normalized input data
    """
    if train:
        MEAN = np.mean(tx, axis=0)
        STD = np.std(tx, axis=0)
    tx -= MEAN
    tx /= STD

    return tx, MEAN, STD


def insert_shift_scalar(tx):
    """Insert shift scalar to the data

    Args:
        tx (np.array): (N, D) input data

    Returns:
        np.array: (N, D + 1) input data with the first col being 1s
    """
    num_samples = tx.shape[0]
    return np.c_[np.ones(num_samples), tx]


def replace_missing_value(tx, strategy, train, VAL_TO_REPLACE_NEG_999):
    """Replacing missing value (-999) with specified strategy

    Args:
        tx (np.array): (N, D) input data
        strategy ('min' | 'most_freq' | 'mean'): strategy fo replace missing value (Note that min is actually min of that feature - 0.001 please see the report.pdf for our reasoning)
        train (bool): True to compute and use new value according to input data False to use the specified VAL_TO_REPLACE_NEG_999 (used on testing data)
        VAL_TO_REPLACE_NEG_999 (np.array): (D, ) array of values to where ith entry is the value to replace missing values in ith col

    Returns:
        np.array, np.array: (N, D) replaced data and (D, ) array of value to replace (to use later on testing data)
    """
    if train:
        VAL_TO_REPLACE_NEG_999 = np.zeros(shape=(tx.shape[1],))

    UNK = -999
    for col in range(tx.shape[1]):
        feature = tx[:, col]
        if train:
            if strategy == "min":
                VAL_TO_REPLACE_NEG_999[col] = np.min(feature[feature != UNK])
                # just to make the missing the minimum and not interfering the existing value
                VAL_TO_REPLACE_NEG_999[col] -= 0.001
            elif strategy == "most_freq":
                values, counts = np.unique(feature[feature != UNK], return_counts=True)
                VAL_TO_REPLACE_NEG_999[col] = values[np.argmax(counts)]
            elif strategy == "mean":
                VAL_TO_REPLACE_NEG_999[col] = np.mean(feature[feature != UNK])

        feature[feature == UNK] = VAL_TO_REPLACE_NEG_999[col]
        tx[:, col] = feature

    return tx, VAL_TO_REPLACE_NEG_999


def build_poly(x, degree):
    """expanding each col for data x, for j=1 to j=degree

    Args:
        x (np.array):  array of data
        degree (int): degree to expand

    Returns:
        np.array: (D, N * degree) expanded data
    """
    expanded_x = x.copy()
    for col_i in range(2, degree + 1):
        expanded_x = np.c_[expanded_x, x ** col_i]
    return expanded_x


def preprocess(
    y,
    tx,
    degree=1,
    strategy="most_freq",
    log=False,
    train=True,
    one_hot_enc=False,
    MEAN=None,
    STD=None,
    VAL_TO_REPLACE_NEG_999=None,
):
    """Preprocess prediction y and input data x with the specified arguments

    Args:
        y (np.array): (N, ) prediction vector
        tx (np.array): (N, D) input data
        degree (int, optional): degree to expand. Defaults to 1.
        strategy ('min' | 'most_freq' | 'mean'): strategy fo replace missing value (Note that min is actually min of that feature - 0.001 please see the report.pdf for our reasoning). efaults to "most_freq".
        log (bool, optional): change y to {0, 1} for preprocessing data for logistic regression and regularized logistic regression. Defaults to False.
        train (bool, optional): [description]. Defaults to True.
        one_hot_enc (bool, optional): flag for doing one hot encoding. Defaults to False.
        MEAN (np.array, optional): (1, D) mean of each feature of input data (specify when train=False). Defaults to None.
        STD (np.array, optional): (1, D) standard deviation of each feature of input data. Defaults to None.
        VAL_TO_REPLACE_NEG_999 (np.array): (D, ) array of values to where ith entry is the value to replace missing values in ith col

    Returns:
        np.array, np.array, (np.array, np.array, np.array): Preprocessed y, x and tuple of mean, std and values used to replace missing values of each column
    """
    tx = np.copy(tx)

    # for logistic regression, we assume y to be in {0, 1}
    if log:
        y[y < 0] = 0

    if one_hot_enc:
        col = 22
        one_hot = one_hot_encode(tx[:, col])
        tx = np.c_[tx[:, :col], tx[:, col + 1 :]]

    tx, VAL_TO_REPLACE_NEG_999 = replace_missing_value(
        tx, strategy, train, VAL_TO_REPLACE_NEG_999
    )
    # in addition, we will add further feature to x
    tx = build_poly(tx, degree)

    # now let's normalize the value in x to mean of 0 and variance of 1
    tx, MEAN, STD = normalize(tx, train, MEAN, STD)

    # let's make a model with shift scalar
    tx = insert_shift_scalar(tx)

    # concatenate the one hot encoded columns
    if one_hot_enc:
        tx = np.c_[tx, one_hot]

    return y, tx, (MEAN, STD, VAL_TO_REPLACE_NEG_999)
