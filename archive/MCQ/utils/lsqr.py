import scipy.sparse.linalg as lng
import numpy as np
import scipy.sparse as spe
import multiprocessing as mp


class CInitializer(object):
    def __init__(self, X, M, K):
        self._B = np.random.randint(K, size=[X.shape[0], M])
        self.M = M
        self.K = K
        self.X = X
        self._shift = np.arange(0, M * self.K, self.K)
        self.one_hot = self.one_hot_encode()

    def one_hot_encode(self):
        # return a sparse array such that:
        # C @ B picks the index of choosed codeword
        N, M = self._B.shape
        total = self._B.size

        shiftB = self._B + self._shift

        # [0*M, 1*M, 2*M, ..., N*M]
        row = (np.arange(0, N)[:, None] * np.ones([N, M], int)).reshape(-1)
        col = shiftB.reshape(-1)
        # return a [N, M*K] matrix with M entry of 1
        return spe.coo_matrix((np.ones(total, bool), (row, col)), shape=(N, M*self.K))

    def lsqr(self, i):
        # for every dim, B @ C[:, i] = X[:, i], so we use B, X[:, i] to solve C[:, i]
        # [N, M*K] @ [M*K] -> [N]
        return lng.lsqr(self.one_hot, self.X[:, i], atol=1e-6, btol=1e-6)[0]

    @property
    def Get(self):
        with mp.Pool(mp.cpu_count()) as pool:
            res = pool.map(self.lsqr, range(self.X.shape[-1]))
        return np.array(res).T

    @property
    def B(self):
        return self._B


class Solver(object):
    def __init__(self, M, K):
        self.M = M
        self.K = K
        self._shift = np.arange(0, M * self.K, self.K)

    def solve(self, X, B):
        self._B = B
        self._X = X
        self.one_hot = self.one_hot_encode()
        with mp.Pool(mp.cpu_count()) as pool:
            res = pool.map(self.lsqr, range(self._X.shape[-1]))
        return np.array(res).T.reshape((self.M, self.K, -1))

    def one_hot_encode(self):
        # return a sparse array such that:
        # C @ B picks the index of choosed codeword
        N, M = self._B.shape
        total = self._B.size
        shiftB = self._B + self._shift

        # [0*M, 1*M, 2*M, ..., N*M]
        row = (np.arange(0, N)[:, None] * np.ones([N, M], int)).reshape(-1)
        col = shiftB.reshape(-1)
        # return a [N, M*K] matrix with M entry of 1
        return spe.coo_matrix((np.ones(total, bool), (row, col)), shape=(N, M * self.K))

    def lsqr(self, i):
        # for every dim, B @ C[:, i] = X[:, i], so we use B, X[:, i] to solve C[:, i]
        # [N, M*K] @ [M*K] -> [N]
        return lng.lsqr(self.one_hot, self._X[:, i], atol=1e-9, btol=1e-9)[0]
