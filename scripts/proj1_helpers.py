# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

# --- Data Loader ---#


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


# --- Data Exporter ---#


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


# --- Data Reporter ---#


def format_result(
    mean_loss_tr,
    mean_loss_te,
    mean_acc_tr,
    mean_acc_te,
    var_acc_tr,
    var_acc_te,
    var_loss_tr=None,
    var_loss_te=None,
    for_spreadsheet=False,
):
    """Format Result for the model report

    Args:
        mean_loss_tr (float): average loss of k-fold losses from training data set
        mean_loss_te (float): average loss of k-fold losses from testing data set
        mean_acc_tr (float): average accuracy of k-fold losses from training data set
        mean_acc_te (float): average accuracy of k-fold losses from training data set
        var_acc_tr (float): variance of accuracy of k-fold losses from training data set
        var_acc_te (float): variance of accuracy of k-fold losses from training data set
        var_loss_tr (float, optional): variance of loss of k-fold losses from training data set. Defaults to None.
        var_loss_te (float, optional): variance of loss of k-fold losses from training data set. Defaults to None.
        for_spreadsheet (bool, optional): flag for formating for copy and pasting ot google spreadsheet. Defaults to False.

    Returns:
        str: formatted result
    """
    if for_spreadsheet:
        return f"Summary: \n\
        {mean_loss_tr}\n\
        {mean_loss_te}\n\
        {mean_acc_tr}\n\
        {mean_acc_te}\n\
        {var_acc_tr}\n\
        {var_acc_te}\n\n\n\
        {var_loss_tr}\n\
        {var_loss_te}\n\
        "
    else:
        return f"Summary: \n\t\
        train loss: {mean_loss_tr}\n\t\
        test loss: {mean_loss_te}\n\t\
        train acc: {mean_acc_tr}\n\t\
        test acc: {mean_acc_te}\n\t\
        train var acc: {var_acc_tr}\n\t\
        test var acc: {var_acc_te}\n\t\n\n\
        train var loss: {var_loss_tr}\n\t\
        test var loss: {var_loss_te}\n\t\
        "
