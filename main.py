import numpy as np  # общие математические и числовые операции
import matplotlib.pyplot as plt  # для построения графиков
import scipy.fft as sc
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


# def dft(function, direction):
#     n = len(function)
#     result = [complex(0, 0)] * n
#
#     for m in range(n):
#         for k in range(n):
#             result[m] += function[k] * np.exp(direction * complex(0, -1) * 2 * np.pi * m * k / n)
#         if direction == -1:
#             result[m] /= n
#
#     return result


def fft(function, direction):  # self, чтобы ссылаться на самих себя
        resultFunction = fft_r(function, direction)  # вызов рекурсивной функции FFT

        if direction == -1:  # если делали прямое преобразование, делим на N каждый полученный элемент после FFT
            valuesLength = len(function)
            for i in range(valuesLength):
                resultFunction[i] /= valuesLength

        return resultFunction


def fft_r(function, direction):
        valuesLength = len(function)

        if valuesLength == 1:  # Если длина вектора равна 1, вернуть function
            return function
        firstHalf = [complex(0, 0)] * (int(valuesLength / 2))  # Подготовка массивов для операции "Бабочка"
        secondtHalf = [complex(0, 0)] * (int(valuesLength / 2))

        # Присвоить Wn значение главного комплексного корня N-й степени из единицы
        Wn = complex(np.cos(2*np.pi/valuesLength), direction * np.sin(2*np.pi/valuesLength))
        w = 1  # Присвоить w = 1
        result = [complex(0, 0)] * valuesLength
        for i in range(int(valuesLength/2)):  # Операция "Бабочка"
            firstHalf[i] = function[i] + function[i + int(valuesLength / 2)]
            secondtHalf[i] = (function[i] - function[i + int(valuesLength / 2)]) * w
            w = w * Wn

        evenResult = \
            fft_r(firstHalf, direction)  # Рекурсивный вызов БПФ для каждой из частей
        oddResult = fft_r(secondtHalf, direction)
        for i in range(valuesLength):  # Объединение результатов
            if i % 2 == 0:
                result[i] = evenResult[int(i / 2)]
            else:
                result[i] = oddResult[int(i / 2)]

        return result

def filter_band_Hamming(function,fc):
    M = len(function)
    h = [0] * M
    for i in range(M):
        if((i-M/2)==0):
            h[i]=2*np.pi*fc
        else:
            h[i] = (np.sin(2*np.pi*fc*(i-M/2)))/(i-M/2)*(0.54 - 0.46 * np.cos((2 * np.pi * i) / M))

    sum = 0
    for i in range(M):
        sum += h[i]
    for i in range(M):
        h[i] /= sum

    result = sc.fft(function)
    result1 = sc.fft(h)

    for i in range(M):
        result[i] *= result1[i]

    res = sc.ifft(result)

    return res


def add_hindrance(function):
    n = len(function)
    result = [0] * n

    for i in range(n):
        result[i] = (function[i]) + np.cos(40 * 2*np.pi * i / N)
    return result



def hff(func, fc):
    x = np.exp((-2)*np.pi*fc)
    a0 = (1 + x) / 2
    a1 = (1 + x) * (-1) / 2
    b1 = x

    N = len(func)
    y = [0] * N
    for i in range(N - 1):
        y[i + 1] = a0 * func[i + 1] + a1 * func[i] + b1 * y[i]
        # print(y[i])
    return y

# def del_freq(func,Hfreq,Lfreq):
#     a =0
#     return a


if __name__ == '__main__':
    N = 1024
    fc =0.14
    arguments = np.arange(0, N) * 2 * np.pi / N
    function = list(map(lambda x: np.sin(3 * x) + np.cos(1*x), arguments))


    # for i in range(N):
    #     function_dft[i] = abs(function_dft[i])

    hindrance_result = add_hindrance(function)
    #
    # function_dft = sc.rfft(hindrance_result)
    # function_dft = del_freq(function_dft)
    # function_res = sc.irfft(function_dft)

    # function_dft = sc.fft(function)

    # print(function_dft)
    # a = function_dft.copy()
    # print(a)
    # print(function_dft)
    # print(len(function_dft))
    # function_dft = fft(function, 1)
    # print(len(function_dft))
    # print(function_dft)
    # result_Hamming = windows_Hamming(function_dft)
    # # res = dft(result_Hamming, -1)
    # res = sc.ifft(result_Hamming)
    # # res = fft(result_Hamming, -1)


    # hindrance_result_dft = dft(hindrance_result, -1)
    # hindrance_result_Hamming = windows_Hamming(hindrance_result_dft)

    # arguments = np.arange(0, 512) * 2 * np.pi / N

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
    # ax_1.grid(True)

    # arguments = np.arange(0, 513) * 2 * np.pi / N

    ax_2.plot(arguments, hindrance_result)
    ax_2.set(title='sin(3x) + cos(x)(иск)')
    # ax_2.scatter(arguments, function, color='orange')
    # ax_2.grid(True)


    # ax_2.plot(arguments, function_dft)
    # ax_2.set(title='sin(3x) + cos(x) ДПФ')
    # ax_2.scatter(arguments, function_dft, color='orange')
    # ax_2.grid(True)
    #

    ax_3.plot(arguments, hff(hindrance_result, fc))
    ax_3.set(title='однопол. ВЧ')
    ax_3.scatter(arguments, hindrance_result, color='orange')
    # ax_3.grid(True)

    # ax_3.plot(arguments, result_Hamming)
    # ax_3.set(title='Полосовой оконный фильтр. Окно Хэмминга')
    # ax_3.scatter(arguments, result_Hamming, color='orange')
    # ax_3.grid(True)

    # arguments = np.arange(0, 1024) * 2 * np.pi / N
    #
    ax_4.plot(arguments, filter_band_Hamming(hindrance_result,fc))
    ax_4.set(title='Однополюсный ВЧ фильтр')
    # ax_4.scatter(arguments, hff(hindrance_result, 0.9), color='orange')
    # ax_4.grid(True)
    #
    # ax_5.plot(arguments, result_Hamming)
    # ax_5.set(title='Hamming')
    # ax_4.scatter(arguments, hff(hindrance_result, 0.9), color='orange')
    # ax_4.grid(True)
    #
    # ax_7.plot(arguments, hindrance_result)
    # ax_7.set(title='sin(3x) + cos(x) (иск)')
    # ax_7.scatter(arguments, hindrance_result, color='orange')
    # ax_7.grid(True)

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
