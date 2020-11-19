import numpy as np

def Reconstruct(C, B):
    M = C.shape[0]
    D = C[0].shape[-1]
    q = np.zeros([B.shape[0], D])
    for i in range(M):
        c = C[i]
        q += c[B[:, i]]
    return q

def QuantizationError(X, C, B):
    return np.sum((X - Reconstruct(C, B)) ** 2, -1)