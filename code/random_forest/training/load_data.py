
import numpy as np

def load_data():
    X_train = np.load('code/random_forest/model_matrices/training/X_train.npy')
    y_train = np.load('code/random_forest/model_matrices/training/y_train.npy')
    X_test = np.load('code/random_forest/model_matrices/test/X_test.npy')
    y_test = np.load('code/random_forest/model_matrices/test/y_test.npy')
    X_val = np.load('code/random_forest/model_matrices/validation/X_validation.npy')
    y_val = np.load('code/random_forest/model_matrices/validation/y_validation.npy')
    
    print("Training matrices shape")
    print(X_train.shape)
    print(y_train.shape)

    print("Test matrices shape")
    print(X_test.shape)
    print(y_test.shape)
    print("Validation matrices shape")
    print(X_val.shape)
    print(y_val.shape)
    return  X_train,y_train,X_test,y_test,X_val,y_val

if __name__ == "__main__":
    load_data()