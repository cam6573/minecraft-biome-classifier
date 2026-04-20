import numpy as np
import os

BASEPATH = "code/adaboost/data/"

def loadData():
        # training
        X_train = np.load(os.path.join(BASEPATH, "training/X.npy"))
        y_train = np.load(os.path.join(BASEPATH, "training/y.npy"))

        # validation
        X_val = np.load(os.path.join(BASEPATH, "validation/X.npy"))
        y_val = np.load(os.path.join(BASEPATH, "validation/y.npy"))

        # test
        X_test = np.load(os.path.join(BASEPATH, "test/X.npy"))
        y_test = np.load(os.path.join(BASEPATH, "test/y.npy"))

        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test