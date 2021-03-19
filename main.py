# This is a sample Python script.

import numpy as np  # общие математические и числовые операции
import matplotlib.pyplot as plt  # для построения графиков
import sympy as sm


def fwht_new(function):
    result = fwht_rec(function)

    values_length = len(function)
    # for i in range(values_length):
    #     result[i] /= values_length

    return result


def fwht_rec(function):
    values_length = len(function)

    if values_length == 1:  # Если длина вектора равна 1, вернуть function
        return function
    first_half = []  # Подготовка массивов для операции "Бабочка"
    second_half = []
    result = []

    for i in range(int(values_length / 2)):  # Операция "Бабочка"
        first_half.append(function[i] + function[i + int(values_length / 2)])
        second_half.append(function[i] - function[i + int(values_length / 2)])

    first_result = fwht_rec(first_half)  # Рекурсивный вызов БПФ для каждой из частей
    second_result = fwht_rec(second_half)

    result = first_result + second_result

    return result


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    N = 16
    arguments = np.arange(0, N) * 2 * np.pi / N
    function = list(map(lambda x: np.sin(3*x) + np.cos(x), arguments))
    # result = fwht(function)
    result = fwht_new(function)
    print(result)
    res = sm.fwht(function)
    print(res)

    fig = plt.figure()
    ax_1 = fig.add_subplot(1, 3, 1)
    ax_4 = fig.add_subplot(1, 3, 2)
    ax_5 = fig.add_subplot(1, 3, 3)

    ax_1.plot(arguments, function)
    ax_1.set(title='sin(3x)+cos(x)')
    ax_1.scatter(arguments, function, color='orange')
    ax_1.grid(True)

    ax_4.plot(arguments, result)
    ax_4.set(title='fwht')
    ax_4.scatter(arguments, result, color='orange')
    ax_4.grid(True)

    ax_5.plot(arguments, res)
    ax_5.set(title='sympy.fwht')
    ax_5.scatter(arguments, res, color='orange')
    ax_5.grid(True)

    plt.show()
