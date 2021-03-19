import numpy as np  # общие математические и числовые операции
import matplotlib.pyplot as plt  # для построения графиков
import sympy as sm  # для функции fwht()


def fwht(function):
    result = fwht_rec(function)

    values_length = len(function)
    for i in range(values_length):
        result[i] /= values_length

    return result


def fwht_rec(function):
    values_length = len(function)

    if values_length == 1:  # Если длина вектора равна 1, вернуть function
        return function
    first_half = []  # Подготовка массивов
    second_half = []

    for i in range(int(values_length / 2)):  # Сложение и вычитание элементов (операция "бабочка")
        first_half.append(function[i] + function[i + int(values_length / 2)])
        second_half.append(function[i] - function[i + int(values_length / 2)])

    first_result = fwht_rec(first_half)  # Рекурсивный вызов БПУ для каждой из частей
    second_result = fwht_rec(second_half)

    result = first_result + second_result  # Объединить результаты

    return result


if __name__ == '__main__':
    N = 16
    arguments = np.arange(0, N) * 2 * np.pi / N
    function = list(map(lambda x: np.sin(3 * x) + np.cos(x), arguments))
    result = fwht(function)
    print(f'fwht: {result}')

    sympy_res = sm.fwht(function)  # для проверки правильности результатов
    values_length = len(function)
    for i in range(values_length):
        sympy_res[i] /= values_length
    print(f'sympy.fwht: {sympy_res}')

    fig = plt.figure()
    ax_1 = fig.add_subplot(1, 3, 1)
    ax_2 = fig.add_subplot(1, 3, 2)
    ax_3 = fig.add_subplot(1, 3, 3)

    ax_1.plot(arguments, function)
    ax_1.set(title='sin(3x) + cos(x)')
    ax_1.scatter(arguments, function, color='orange')
    ax_1.grid(True)

    ax_2.plot(arguments, result)
    ax_2.set(title='fwht')
    ax_2.scatter(arguments, result, color='orange')
    ax_2.grid(True)

    ax_3.plot(arguments, sympy_res)
    ax_3.set(title='sympy.fwht')
    ax_3.scatter(arguments, sympy_res, color='orange')
    ax_3.grid(True)

    plt.show()
