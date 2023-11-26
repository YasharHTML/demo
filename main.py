import numpy as np
from math import *
from numpy.linalg import solve


def PartialPivoting(A, B):
    n = len(B)
    x = np.zeros(n, dtype=float)
    S = np.array([max(abs(A[i])) for i in range(0, n)], dtype=float)
    L = [i for i in range(0, n)]

    # forward elemination
    for k in range(n - 1):
        Ratios = np.array([abs(A[m][k] / S[m]) for m in L[k:]], dtype=float)

        pivot = Ratios.argmax() + k
        L[k], L[pivot] = L[pivot], L[k]

        for i in L[k + 1 : :]:
            factor = A[i][k] / A[L[k]][k]
            B[i] -= factor * B[L[k]]

            for j in range(n):
                A[i][j] -= factor * A[L[k]][j]

    # backward subsitution
    ind = n - 1
    for i in L[::-1]:
        sum = B[i]
        for j in range(n - 1, ind, -1):
            sum -= A[i][j] * x[j]
        x[ind] = sum / A[i][ind]
        ind -= 1

    return x
