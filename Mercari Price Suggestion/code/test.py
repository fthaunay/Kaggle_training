import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix
import time
from sklearn.linear_model import Ridge


def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def main():
    train = pd.read_csv("../input/train.tsv", sep='\t')
    test = pd.read_csv("../input/test.tsv", sep='\t')


    X_train = np.loadtxt("x_train.csv", delimiter=",")
    X_test = np.loadtxt("x_test.csv", delimiter=",")

    y_train = np.log1p(train['price'].as_matrix())



    regr = Ridge(alpha=1.0)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    submission = pd.DataFrame(test[["test_id"]])
    submission['price'] = np.expm1(y_pred)
    submission.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    main()