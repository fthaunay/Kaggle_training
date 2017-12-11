import gc
import time
import numpy as np
from scipy.sparse import csr_matrix

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])



def main():

    start_time = time.time()
    X_train = np.loadtxt("code/X_train.csv", delimiter=",")
    X_test = np.loadtxt("code/X_train.csv", delimiter=",")

    X_train_sparse = csr_matrix(X_train)
    X_test_sparse = csr_matrix(X_test)

    save_sparse_csr("X_train_sparse", X_train_sparse)
    save_sparse_csr("X_test_sparse", X_test_sparse)


if __name__ == '__main__':
    main()