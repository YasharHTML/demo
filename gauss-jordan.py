import numpy as np

A = np.array([[3, 2, -1], [9, 7, -5], [-6, 6, 3]], dtype=float)


def GaussJordan(A):
    size = len(A)
    A[0] /= A[0][0]
    for i in range(size):
        for j in range(i + 1, size):
            coefficient = A[j][i] / A[i][i]
            A[j] -= coefficient * A[i]

    A[size - 1] /= A[size - 1][-1] 
    
    for i in range(size - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            coefficient = A[j][i] / A[i][i]
            A[j] -= coefficient * A[i]

GaussJordan(A)
print(A)