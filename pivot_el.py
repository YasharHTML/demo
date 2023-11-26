import matplotlib.pyplot as plt
import numpy as np
from math import *
from numpy.linalg import inv 

def pivot(A, B):
    n = len(B)
    X = np.zeros(n, dtype=float)
    S = np.array([np.max(abs(i)) for i in A], dtype=float)
    L = np.array([i + 1 for i in range(n)], dtype=float)
    
    for k in range(n - 1):
        if fabs(A[k,k]) <= 1.e-12:
            for i in range(k + 1, n):
                if A[k, i] > 1.e-12:
                    A[[k, i]] = A[[i, k]]
                    B[[k, i]] = B[[i, k]]
                    break                
        for i in range(k + 1, n):
            factor = A[i,k] / A[k,k]
            for j in range(k, n):
                A[i,j] = A[i,j] -  factor * A[k,j]
            B[i] = B[i] - factor * B[k]                  
    #backward substitution
    X[n - 1] = B[n - 1] / A[n - 1, n - 1]
    for i in range(n-2, -1, -1):
        sum_ = 0
        for j in range(i+1, n):
            sum_ += A[i,j] * X[j]
        X[i] = (B[i] - sum_) / A[i,i]
        
    return X
A = np.array([[0, 4, 6], [4, -5, 6], [3, 1, -2]], dtype = float)
B = np.array([18, 24, 4], dtype = float)

print(f"Augmented matrix of the system of linear equations:\n{A}")
print(f"B matrix of the system of linear equations:\n {B}")

print(F"Checking the roots: \n{np.linalg.solve(A,B)}")

print(f"Solution of system: \n{pivot(A,B)}")