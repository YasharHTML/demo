import numpy as np

A = np.array([[6, -2, 2, 4], [12, -8, 6, 10], [3, -13, 9, 3], [-6, 4, 1, -18]], dtype=float)
B = np.transpose(np.array([16, 26, -19, -34], dtype=float))

def GaussianElimination(A, B):
  size = len(B)
  for i in range(size):
    for j in range(i + 1, size):
      coefficient = A[j][i] / A[i][i]
      A[j] -= coefficient * A[i]
      B[j] -= coefficient * B[i]
  
  x = np.zeros(size)
  for i in range(size - 1, -1, -1):
    temp = B[i]
    for j in range(i + 1, size):
      temp -= A[i][j] * x[j]
    x[i] = temp / A[i][i]

  return A, B, x

GaussianElimination(A, B)