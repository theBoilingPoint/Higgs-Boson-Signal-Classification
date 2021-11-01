import numpy as np
from proj1_helpers import *

from implementations import *
from cross_validation import *
from preprocess import preprocess

SEED = 1
np.random.seed(SEED)

DATA_TRAIN_PATH = "../data/train.csv"
Y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

k_fold = 5
degree = 2
one_hot_enc = True
strategy = "most_freq"

log = True
threshold = 0.5

y, tx_train, (MEAN, STD, VAL_TO_REPLACE_NEG_999) = preprocess(
    Y,
    tX,
    degree=degree,
    strategy=strategy,
    one_hot_enc=one_hot_enc,
    train=True,
    log=log,
)

initial_w = np.zeros(tx_train.shape[1])
lambda_ = 1
gamma = 1e-6
each_iters = 1000

compute_loss = lambda y, x, w: compute_loss_logistic_regression(
    y, x, w
) + loss_reg_logistic_regression(lambda_, w)

weight = initial_w
loss_tr = None

for i in range(1, 15):
    weight, loss_tr = reg_logistic_regression(
        y,
        tx_train,
        initial_w=weight,
        max_iters=each_iters,
        lambda_=lambda_,
        gamma=gamma,
        verbose=1,
    )

    acc_tr = compute_acc(y, tx_train, weight, threshold, log)

    print(f"Iteration {i * each_iters}: loss_te={loss_tr}, acc_tr={acc_tr}")

    with open(f"../weights/logistic_regression_models_{i * each_iters}.npy", "wb") as f:
        np.save(f, weight)


DATA_TEST_PATH = "../data/test.csv"
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
_, tX_test, _ = preprocess(
    _,
    tX_test,
    degree=degree,
    strategy=strategy,
    one_hot_enc=one_hot_enc,
    log=log,
    train=False,
    MEAN=MEAN,
    STD=STD,
    VAL_TO_REPLACE_NEG_999=VAL_TO_REPLACE_NEG_999,
)

OUTPUT_PATH = f"../output/logistic_regression_N_Md_D2_1H_lam1_gam1e-06_trainall_{i * each_iters}.csv"
y_pred = make_prediction(tX_test, weight, threshold=threshold, log=log, test=True)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
