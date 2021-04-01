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

def add_hindrance(function):
    n = len(function)
    result = [0] * n

    for i in range(n):
        result[i] = function[i] + random.random()

    return result



if __name__ == '__main__':
    N = 16
    arguments = np.arange(0, N) * 2 * np.pi / N

    function = list(map(lambda x: np.sin(3 * x) + np.cos(x), arguments))
    function_dft = dft(function, -1)
    result_Hamming = windows_Hamming(function_dft)

    hindrance_result = add_hindrance(function)
    hindrance_result_dft = dft(hindrance_result, -1)
    hindrance_result_Hamming = windows_Hamming(hindrance_result_dft)

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
    ax_1.scatter(arguments, function, color='orange')
    ax_1.grid(True)

    ax_2.plot(arguments, function_dft)
    ax_2.set(title='sin(3x) + cos(x) ДПФ')
    ax_2.scatter(arguments, function_dft, color='orange')
    ax_2.grid(True)

    ax_3.plot(arguments, result_Hamming)
    ax_3.set(title='Полосовой оконный фильтр. Окно Хэмминга')
    ax_3.scatter(arguments, result_Hamming, color='orange')
    ax_3.grid(True)

    ax_7.plot(arguments, hindrance_result)
    ax_7.set(title='sin(3x) + cos(x) (иск)')
    ax_7.scatter(arguments, hindrance_result, color='orange')
    ax_7.grid(True)

    ax_8.plot(arguments, hindrance_result_dft)
    ax_8.set(title='sin(3x) + cos(x) ДПФ (иск)')
    ax_8.scatter(arguments, hindrance_result_dft, color='orange')
    ax_8.grid(True)

    ax_9.plot(arguments, hindrance_result_Hamming)
    ax_9.set(title='Полосовой оконный фильтр. Окно Хэмминга (иск)')
    ax_9.scatter(arguments, hindrance_result_Hamming, color='orange')
    ax_9.grid(True)


    plt.show()
