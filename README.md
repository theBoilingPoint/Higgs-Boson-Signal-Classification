# Higgs Boson - EPFL ML Project 1

Machine Learning Project 1

## Team Members

Naravich Chutisilp - naravich.chutisilp@epfl.ch<br>
Xinran Tao - xinran.tao@epfl.ch<br>
Dangyu Wang - danyu.wang@epfl.ch<br>

## Quick Start

Clone this reponsitory<br>
Train the model with

```
cd project1/scripts/
python3 run.py
```

_Note_ this repo have `lfs` files, don't forget to clone or pull with `lfs`<br>
_Note2_ We actually have the saved model we used in `./weights/logistic_regression_models_14000.npy`. (Thus, instead of running run.py one can use `np.load` and get the weight that we have trained<br>
_Note3_ We actually have the actual submission we submitted in `./output/logistic_regression_N_Md_D2_1H_lam1_gam1e-06_trainall_14000.csv`<br>

## Objective

Classify classify events as the result of Higgs Boson (signal) or others (background)

## Result

We ranked 163rd out of 228 teams with accuracy of 0.804 and F1 score of 0.694

## Structure of this project

```
project
│   README.md
|   report.pdf
│   project1_description.pdf
│
├───data
│      train.csv # the data for training
│      test.csv # data for submission
│
├───output # directory storing intermediate output during model exploration
│
├───weights # directory storing weights to resume training in the future
│
└───scripts
      #--- scripts for running model ---#
      build_poly.py # utilities function for polynomial expansion
      preprocess.py # utilities functions for data preprocessing
      cross_validation.py # utilities function for cross validation
      implementations.py # 6 model implementations and their intermediate functions
      proj1_helper.py # helper functions such as reading csv data, creating submission files and reporting model results
      project1.ipynb # models trained with their optimal hyperparameter and reported result on training set
      run.py # script for training and making prediction of our submitted model on aicrowd

      #--- scripts for hyperparameter exploration (see summary result [here](https://docs.google.com/spreadsheets/d/168y-Jz0eW68siLYuDIBLWPEUJdhqdbj5nAuehSxlt1Q/edit?usp=sharing)) ---#
      least_square_GD_SGD_exploration.ipynb
      least_square_ridge_logistic_reg_logisric_exploration.ipynb
```

## Data Preprocessing

Use funtion `preprocess` in `/scripts/preprocess.py` which including some options for cleaning and manipulating the data.

1. (for logistic regression and regularized logistic regression) Make y to be {0, 1} instead of {-1, 1}
2. Do one hot encoding on `PRI_jet_num` col because it is the only categorical feature
3. Replace missing values (-999) with one the three strategies (most frequent, mean, min - 0.001)
4. Normalize the data by subtracting mean and dividing it by variance (z-normalization) on numerical features
5. Do polynomial expansion on numerical featuers
6. Add shift scalar col (col with 1)

Note: for testing data normalization and missing data replacement will use the corresponding values obtained from training data (using flag `train`=`True` to obtain the mean, standard deviation, and values to replace -999, then pass these values on the `preprocess` with `train`=`False` when preprocess the testing data)

## Models

_6_ models were implemented in `/scripts/implementations.py` including

1. Linear regression with gradient descent
2. Linear regression with stochastic gradient descent
3. Least square
4. Ridge regression
5. Logistic regression
6. Regularized logistic regression

## Cross validation
In order to do k-fold cross-validation, one can use functions in `/scripts/cross_validation.py` (all the code for doing cross validation on each model can be found in `/scripts/project1.ipynb`).
For example, doing 5-fold cross validation for regularized logistic regression can be
```
from implementations import *
from cross_validation import *
from preprocess import *

# Preprocess the data
y, tx_train = preprocess(Y, tX, degree=2, strategy='most_freq', log=True, one_hot_enc=True)

# Prepare hyperparameters
initial_w = np.zeros(tx_train.shape[1])
k_indices = build_k_indices(y, k_fold, seed)

# Create lambda function to pass to cross_validation
trainer = lambda y, x: reg_logistic_regression(y, x, lambda_, initial_w, max_iters, gamma, verbose=0)
compute_loss = lambda y, x, w: compute_loss_logistic_regression(y, x, w) + loss_reg_logistic_regression(lambda_, w)

acc_tr, acc_te, loss_tr, loss_te, weight = cross_validation(y, tx_train, k_indices, k, trainer, compute_loss=compute_loss, threshold=0.5, log=True)
```
