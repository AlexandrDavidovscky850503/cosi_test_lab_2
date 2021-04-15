import numpy as np  # общие математические и числовые операции
import matplotlib.pyplot as plt  # для построения графиков
import random
#import sympy as sm  # для функции fwht()

def windows_Hamming(function):
    values_length = len(function)
    w = [0]*values_length
    result = [0] * values_length
    for i in range(values_length):
        w[i] = 0.54 - 0.46 * np.cos(2 * np.pi / values_length) * i
    for i in range(values_length):
        result[i] = function[i] * w[i]
    return result


def dft(function, direction):
    n = len(function)
    result = [complex(0, 0)] * n

    for m in range(n):
        for k in range(n):
            result[m] += function[k] * np.exp(direction * complex(0, -1) * 2 * np.pi * m * k / n)
        if direction == -1:
            result[m] /= n

    return result

# def add_hindrance(function):
#     n = len(function)
#     result = [0] * n
#
#     for i in range(n):
#         result[i] = function[i] + random.random()
#
#     return result

def add_hindrance(function):
    n = len(function)
    result = [0] * n

    for i in range(n):
        result[i] = (np.sin(3 * 2 * np.pi * i / N) + np.cos(2 * np.pi * i / N)) + np.cos(40 * 2*np.pi * i / N)

    return result


def hff(func, x):
    a0 = (1 + x) / 2
    a1 = (1 + x) * (-1) / 2
    b1 = x

    N = len(func)
    y = [0] * N
    for i in range(N  - 1):
        y[i + 1] = a0 * func[i + 1] + a1 * func[i] + b1 * y[i]
        print(y[i])

    return y


# def hff(func, x):
#     a0 = 0.15
#     b1 = 0.85
#
#     N = len(func)
#     y = [0] * N
#     for i in range(N - 1):
#         y[i + 1] = a0 * func[i + 1] + b1 * y[i]
#
#     return y

# def hff(func, x):
#     a0 = (1 - x) ** 4
#     b1 = 4 * x
#     b2 = -6 * x**2
#     b3 = 4 * x ** 3
#     b4 = -x ** 4
#
#     N = len(func)
#     y = [0] * (N)
#     for i in range(N - 4):
#         y[i + 4] = a0 * func[i + 4] + b1 * y[i + 3] + b2 * y[i + 2] + b3 * y[i + 1] + b4 * y[i]
#
#     return y

# def hff(func, X):
#
#     y = np.zeros(N)
#
#     for j in range(N):
#         y[j] = (np.power((1 - X), 4) * func[j]) + (4 * X * y[j - 1]) + (-6 * np.power(X, 2) * y[j - 2]) \
#                + (4 * np.power(X, 3) * y[j - 3]) + (-np.power(X, 4) * y[j - 4])
#
#     return y

if __name__ == '__main__':
    N = 1024
    arguments = np.arange(0, N) * 2 * np.pi / N

    function = list(map(lambda x: np.sin(3 * x) + np.cos(x), arguments))
    # function_dft = dft(function, -1)
    # result_Hamming = windows_Hamming(function_dft)
    #
    hindrance_result = add_hindrance(function)
    # hindrance_result_dft = dft(hindrance_result, -1)
    # hindrance_result_Hamming = windows_Hamming(hindrance_result_dft)

    fig = plt.figure()
    ax_1 = fig.add_subplot(3, 3, 1)
    ax_2 = fig.add_subplot(3, 3, 2)
    ax_3 = fig.add_subplot(3, 3, 3)

    ax_4 = fig.add_subplot(3, 3, 4)
    ax_5 = fig.add_subplot(3, 3, 5)
    ax_6 = fig.add_subplot(3, 3, 6)

    ax_7 = fig.add_subplot(3, 3, 7)
    ax_8 = fig.add_subplot(3, 3, 8)
    ax_9 = fig.add_subplot(3, 3, 9)

    ax_1.plot(arguments, function)
    ax_1.set(title='sin(3x) + cos(x)')
    # ax_1.scatter(arguments, function, color='orange')
    ax_1.grid(True)

    # ax_2.plot(arguments, function_dft)
    # ax_2.set(title='sin(3x) + cos(x) ДПФ')
    # ax_2.scatter(arguments, function_dft, color='orange')
    # ax_2.grid(True)
    #
    # ax_3.plot(arguments, result_Hamming)
    # ax_3.set(title='Полосовой оконный фильтр. Окно Хэмминга')
    # ax_3.scatter(arguments, result_Hamming, color='orange')
    # ax_3.grid(True)

    ax_4.plot(arguments, hff(hindrance_result, 0.85))
    ax_4.set(title='Полосовой оконный фильтр. Окно Хэмминга')
    # ax_4.scatter(arguments, hff(hindrance_result, 0.9), color='orange')
    ax_4.grid(True)

    ax_7.plot(arguments, hindrance_result)
    ax_7.set(title='sin(3x) + cos(x) (иск)')
    # ax_7.scatter(arguments, hindrance_result, color='orange')
    ax_7.grid(True)

    # ax_8.plot(arguments, hindrance_result_dft)
    # ax_8.set(title='sin(3x) + cos(x) ДПФ (иск)')
    # ax_8.scatter(arguments, hindrance_result_dft, color='orange')
    # ax_8.grid(True)
    #
    # ax_9.plot(arguments, hindrance_result_Hamming)
    # ax_9.set(title='Полосовой оконный фильтр. Окно Хэмминга (иск)')
    # ax_9.scatter(arguments, hindrance_result_Hamming, color='orange')
    # ax_9.grid(True)


    plt.show()
