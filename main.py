# This is a sample Python script.

import numpy as np  # общие математические и числовые операции
import matplotlib.pyplot as plt  # для построения графиков
import sympy as sm


def sequencyReordering(inputData):
    inputLength = len(inputData)

    if isPowerOfTwo(inputLength):
        length = inputLength
        bitsInLenght = int(np.log2(length))
    else:
        bitsInLenght = np.log2(inputLength)
        length = 1 << bitsInLenght

    data = inputData[0:length]
    for i in range(length):
        data[i] = inputData[grayToBinary(bitsRevers(i, bitsInLenght))]

    return data


def isPowerOfTwo(n):
    return n > 1 and (n & (n - 1)) == 0


def fwht(inputFunction):
    inputLenght = len(inputFunction)

    if isPowerOfTwo(inputLenght):
        length = inputLenght
        bitsInLength = int(np.log2(length))
    else:
        bitsInLength = np.log2(inputLenght)
        length = 1 << bitsInLength

    data = sequencyReordering(inputFunction)

    for ldm in range(bitsInLength, 0, -1):
        m = 2 ** ldm
        mh = int(m / 2)
        for k in range(mh):
            for r in range(0, length, m):
                u = data[r + k]
                v = data[r + k + mh]

                data[r + k] = u + v
                data[r + k + mh] = u - v


    data = list(np.divide(data, length))

    return data

def bitsRevers(n, numberOfBits):
    reverseBits = 0

    for i in range(numberOfBits):
        next_bit = n & 1
        n >>= 1

        reverseBits <<= 1
        reverseBits |= next_bit

    return reverseBits


def grayToBinary(num):
    mask = num
    while mask != 0:
        mask >>= 1
        num ^= mask

    return num


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
    # ax_2 = fig.add_subplot(3, 3, 2)
    # ax_3 = fig.add_subplot(3, 3, 3)
    ax_4 = fig.add_subplot(1, 3, 2)
    ax_5 = fig.add_subplot(1, 3, 3)
    # ax_6 = fig.add_subplot(3, 3, 9)
    #
    ax_1.plot(arguments, function)
    ax_1.set(title='sin(3x)+cos(x)')
    ax_1.scatter(arguments, function, color='orange')
    ax_1.grid(True)
    #
    ax_4.plot(arguments, result)
    ax_4.set(title='fwht')
    ax_4.scatter(arguments, result, color='orange')
    ax_4.grid(True)
    #
    # ax_2.plot(arguments, correlation)
    # ax_2.set(title='Корреляция')
    # ax_2.scatter(arguments, correlation, color='orange')
    # ax_2.grid(True)
    #
    # ax_3.plot(arguments, convolution)
    # ax_3.set(title='Свертка')
    # ax_3.scatter(arguments, convolution, color='orange')
    # ax_3.grid(True)
    #
    ax_5.plot(arguments, res)
    ax_5.set(title='sympy.fwht')
    ax_5.scatter(arguments, res, color='orange')
    ax_5.grid(True)
    #
    # ax_6.plot(arguments, convolutionWithFastFourierTransform)
    # ax_6.set(title='Свертка с БПФ')
    # ax_6.scatter(arguments, convolutionWithFastFourierTransform, color='orange')
    # ax_6.grid(True)
    #
    plt.show()
