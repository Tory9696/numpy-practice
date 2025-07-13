import numpy as np
from afxres import AFX_IDD_REPLACE
from numba.cuda.printimpl import print_item
from textdistance import Matrix

"""
# Задача: найти значения матриц для выражения 0.5A + 2B - E
A = np.array([[-2, 4, 0],
              [2, -5, 3],
              [2, 0, 6]])
B = np.array([[-2, -4, 6],
              [1, 0, 3],
              [1, 2, -5]])
E = np.eye(3)
res = np.sum(0.5 * A + 2 * B - E)
print(np.sum(res))"""

"""
# Задача: найти сумму AB и BA
A = np.array([[3],
              [4],
              [2]])
B = np.array([[5, -2, 3]]) # Но можно использовать reshape для явного преобразования строки в матрицу

res = np.sum(A@B) + np.sum(B@A) # Символ @ это матричное умножение
print(res)"""

"""# Задача: найти определитель матрицы третьего порядка
A = np.array([[3, 4, -5],
              [5, 5, 3],
              [2, 1, 8]])
print(round(np.linalg.det(A)))"""

"""# Задача: решить уравнение (получить корень или сумму корней, если их несколько)

import sympy as sp # Библиотека для символьных вычислений
from sympy import Matrix # Для создания матрицы
x = sp.symbols("x") # Создание символа x, котрый будет использоваться в матрице
A = Matrix([[5, -3, x], [1, 1, -2], [2, x+2, -1]]) # Матрица
det = A.det() # Определитель
print(sum(sp.solve(det))) # Крутейший метод solve() для решения уравнений"""

"""# Задача: найти определитель матрицы четвёртого порядка
A = np.array([[2, 1, 5, 1],
              [3, 2, 1, 2],
              [1, 2, 3, -4],
              [1, 1, 5, 1]])
print(round(np.linalg.det(A)))"""

"""# Задача: найти определитель матрицы пятого порядка
A = np.array([[7, 8, 5, 5, 3],
              [10, 11, 6, 7, 5],
              [5, 3, 6, 2, 5],
              [6, 7, 5, 4, 2],
              [7, 10, 7, 5, 0]])
print(round(np.linalg.det(A)))"""


"""# Задача: найти элемент третьей строки и второго столбца обратной матрицы четвёртого порядка
A = np.array([[0, 0, 1, 1],
              [0, 3, 1, -7],
              [2, 7, 6, 1],
              [1, 2, 2, 1]])
print(np.linalg.inv(A)[2, 1])"""

"""# Задача: решить матричное уравнение: [[3, -1], [5, -2]] * X * [[5, 6],[7, 8]] = [[14, 16], [9, 10]]
A = np.array([[3, -1], [5, -2]])
B = np.array([[5, 6], [7, 8]])
C = np.array([[14, 16], [9, 10]])
# Алгоритм решения, как обычные уравнения, но тут обратные матрицы слева равенства и исходная справа
# X = A ** -1 * C * B ** -1
a_inv = np.linalg.inv(A)
b_inv = np.linalg.inv(B)

x = a_inv@C@b_inv # Сначала A на C, потом результат на B
# Можно использовать np.dot для матричного умножения, но @ более лаконичен

print(round(x.sum()))"""

"""# Задача: найти ранг матрицы
from numpy.linalg import matrix_rank

A_rank = [[1, 1, 2, 3, -1],
          [2, -1, 0, -4, -5],
          [-1, -1, 0, -3, -2],
          [6, 3, 4, 8, -3]]

print(matrix_rank(A_rank))"""

"""# Задача: получить транспонированную и инверт. матрицу A, её детерминант и произведение AA**-1, и сумму элементов
A = np.array([[1, 2, 3],
              [4, 2, 1],
              [1, 0, 1]])
# print(A.transpose()) #
# a_inv = np.linalg.inv(A)
# print(np.sum(A@a_inv))
# print(round(np.linalg.det(A)))
# print(np.sum(np.linalg.inv(A)))

# Задача: получить сумму элементов главной диагонали новой матрицы res
B = np.array([[1, 2, 1],
              [1, -1, 2],
              [1, 2, 1]])
E = np.identity(3)
res = (-1) * A + 2 * B + (-3) * E
print(res)
print((-2) + (-7) + (-2))

# Задача: умножить -3 на матрицу C, получить детерминант C
C = np.array([[2, 2,],
              [1, 4]])
# print(np.sum(-3 * C))
# print(np.linalg.det(C))

# Задача: получить транспонированную матрицу D
D = np.array([[3, 1],
              [2, 1],
              [-1, 2]])
# print(D.transpose())

# Задача: получить детерминант G
G = np.array([[1, 1, 2, 3],
              [4, 2, 1, 3],
              [2, 1, 2, 3],
              [1, 2, 5, 4]])
# print(round(np.linalg.det(G)))

# Задача: получить все возможные парные произведения матриц ABCDG и сумму элементов [0, 1]
# print((A@B)[0, 1] + (A@D)[0, 1] + (B@A)[0, 1] + (B@D)[0, 1] + (D@C)[0, 1])"""

"""# Задача: решить СЛАУ
# 4x_1 + 2x_2 - x_3 = 0
# x_1 + 2x_2 + x_3 = 1
# x_2 - x_3 = -3

A = np.array([[4, 2, -1],
              [1, 2, 1],
              [0, 1, -1]])
B = np.array([0, 1, -3])
print(np.sum(np.linalg.solve(A, B)))"""

"""# Задача: решить СЛАУ и найти сумму неизвестных переменных
A = np.array([[2, 3, 11, 5],
              [1, 1, 5, 2],
              [3, 3, 9, 5],
              [2, 1, 3, 2],
              [1, 1, 3, 4]])
B = np.array([2, 1, -2, -3, -3])
x = np.linalg.lstsq(A, B, rcond=None)[0]
print(f'x1 = {x[0]}, x2 = {x[1]}, x3 = {x[2]}, x4 = {x[3]}')
print(round(np.sum(x)))"""

"""A = np.array([[1, 4, -1],
              [0, 5, 4],
              [3, -2, 5]])
A_hide = np.array([[1, 4, -1, 6],
                   [0, 5, 4, -20],
                   [3, -2, 5, -22]])
a_det = np.linalg.det(A)
a_rank = np.linalg.matrix_rank(A)
a_hide_rank = np.linalg.matrix_rank(A_hide)
B = np.array([6, -20, -22])
x = np.linalg.lstsq(A, B, rcond=None)[0]
print(a_rank)"""

"""A = np.array([[4, -9, 5],
              [7, -4, 1],
              [3, 5, -4]])
A_hide = np.array([[4, -9, 5, 1],
                   [7, -4, 1, 11],
                   [3, 5, -4, 5]])
a_det = np.linalg.det(A)
a_rank = np.linalg.matrix_rank(A)
a_hide_rank = np.linalg.matrix_rank(A_hide)
print(a_det, a_rank, a_hide_rank) # Не совместная, det = 0"""

"""A = np.array([[1, 7, -3],
              [3, -5, 1],
              [3, 4, -2]])
B = np.array([0, 0, 0])
print(np.sum(np.linalg.solve(A, B)))
# print(np.linalg.matrix_rank(A)) # Однородная система и решение 0"""

"""A = np.array([[3, -2, 1],
              [3, 3, -5],
              [6, 1, -4]])
B = np.array([0, 0, 0])
# print(np.sum(np.linalg.solve(A, B)))
print(np.linalg.matrix_rank(A))
print(np.linalg.det(A)) # Однородная система, имеет бесконечно много решений, т.к. r < n (n - неизв.перемен.)"""

"""A = np.array([[-1, 0, -1, -2, 2],
              [2, -1, 1, -2, 2],
              [1, -2, 0, -1, -1],
              [2, 1, 2, 1, 1]])
a_rank = np.linalg.matrix_rank(A)
B = np.argmax([1, -1, 0, -2])
print(a_rank)"""

