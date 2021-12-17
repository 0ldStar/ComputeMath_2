import numpy as np

A = [[70, 2, 40, 0, 21],
     [9, 25, 0, 50, -14],
     [25, 0, 40, -1, 13],
     [0, 32, 0, 90, 21]]

A1 = [[70, 2, 40, 0],
      [9, 25, 0, 50],
      [25, 0, 40, -1],
      [0, 32, 0, 90]]

A2 = [21, -14, 13, 21]


def matrix_max_row(matrix, n):
    max_element = matrix[n][n]
    max_row = n
    for i in range(n + 1, len(matrix)):
        if abs(matrix[n][i]) > abs(max_element):
            max_element = matrix[n][i]
            max_row = i
        if max_row != n:
            matrix[n], matrix[max_row] = matrix[max_row], matrix[n]


def Gauss(matrix):
    n = len(matrix)
    x = np.zeros(n)

    for k in range(n - 1):
        matrix_max_row(matrix, k)
        for i in range(k + 1, n):
            div = matrix[i][k] / matrix[k][k]
            matrix[i][-1] -= div * matrix[k][-1]
            for j in range(k, n):
                matrix[i][j] -= div * matrix[k][j]
    if is_singular(matrix):
        raise RuntimeError("endless solution")
    for k in range(n - 1, -1, -1):
        x[k] = (matrix[k][-1] - sum([matrix[k][j] * x[j] for j in range(k + 1, n)])) / matrix[k][k]
    return x


def is_singular(matrix):
    for i in range(len(matrix)):
        if not matrix[i][i]:
            return True
        return False


def seidel(matrix, b, eps):
    n = len(matrix)
    x = np.zeros(n)

    converge = False
    while not converge:
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(matrix[i][j] * x_new[j] for j in range(i))
            s2 = sum(matrix[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / matrix[i][i]
        converge = np.sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(n))) <= eps
        x = x_new

    return x


def det2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def minor(matrix, i, j):
    tmp = [row for k, row in enumerate(matrix) if k != i]
    tmp = [col for k, col in enumerate(zip(*tmp)) if k != j]
    return tmp


def det(matrix):
    size = len(matrix)
    if size == 2:
        return det2(matrix)

    return sum((-1) ** j * matrix[0][j] * det(minor(matrix, 0, j))
               for j in range(size))


def kramer(les):
    n = len(les)
    tmp = list(zip(*les))
    b = tmp[-1]
    del tmp[-1]

    delta = det(tmp)
    if delta == 0:
        raise RuntimeError("solution is absent")
    result = []
    for i in range(n):
        a = tmp[:]
        a[i] = b
        result.append(det(a) / delta)
    return result


def print_result(x):
    for i in x:
        print("%0.5f " % i, end='')
    print()


print("Kramer: ")
print_result(kramer(A))

print("Gauss: ")
print_result(Gauss(A))

print("Seidel: ")
print_result(seidel(A1, A2, 10e-5))

print("NumPy: ")
print_result(np.linalg.solve(A1, A2))
