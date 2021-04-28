import numpy as np  # общие математические и числовые операции
import matplotlib.pyplot as plt  # для построения графиков


def filter_band_Hamming(function, fc, fc1):
    N = len(function)
    M = int(N/10)
    print(M)
    h = [0] * M
    for i in range(M):
        if (i-M/2) == 0:
            h[i] = 2*np.pi*fc
        else:
            h[i] = (np.sin(2*np.pi*fc*(i-M/2)))/(i-M/2)
        h[i] *= (0.54 - 0.46 * np.cos((2 * np.pi * i) / M))

    h1 = [0] * M
    for i in range(M):
        if (i - M / 2) == 0:
            h1[i] = 2 * np.pi * fc1
        else:
            h1[i] = (np.sin(2 * np.pi * fc1 * (i - M / 2))) / (i - M / 2)

        h1[i] *= (0.54 - 0.46 * np.cos((2 * np.pi * i) / M))

    sum = 0
    for i in range(M):
        sum += h[i]
    if sum:
        for i in range(M):
            h[i] /= sum

    sum1 = 0
    for i in range(M):
        sum1 += h1[i]
    if sum1:
        for i in range(M):
            h1[i] /= sum1

    for i in range(M):
        h1[i] = h1[i]*(-1)

    h1[int(M/2)] = h1[int(M/2)] + 1

    h2 = [0] * M
    for i in range(M):
        h2[i] = h1[i]+h[i]

    for i in range(M):
        h2[i] = h2[i]*(-1)

    h2[int(M/2)] = h2[int(M/2)] + 1

    res = [0] * len(function)
    for i in range(len(function)):
        if i > M:
            temp = 0
            for j in range(M):
                temp += function[i-j]*h2[j]
            res[i] = temp

    return res


def add_hindrance(function):
    n = len(function)
    result = [0] * n

    for i in range(n):
        result[i] = (function[i]) + np.cos(75 * 2 * np.pi * i / N)
        # if i > int(n / 3):
        #     if i < int(n / 3 + 0.2*n):
        #         result[i] = (function[i]) + np.cos(75 * 2 * np.pi * i / N)
        #     else:
        #         result[i] = (function[i])
        # else:
        #     result[i] = (function[i])

    return result


def hff(func, fc):
    x = np.exp((-2)*np.pi*fc)
    a0 = (1 + x) / 2
    a1 = (-1 - x) / 2
    b1 = x

    N = len(func)
    y = [0] * N
    for i in range(N):
        if i > 1:
            y[i] = a0 * func[i] + a1 * func[i-1] + b1 * y[i-1]

    return y


if __name__ == '__main__':
    N = 512
    fc_hff = 0.097
    fc = 0
    fc1 = 0.05

    # fc = 0.097
    # fc1 = 0.41

    arguments = np.arange(0, N) * 2 * np.pi / N
    arguments2 = np.arange(0, N / 2 + 1)
    function = list(map(lambda x: np.sin(3 * x) + np.cos(x), arguments))
    hindrance_result = add_hindrance(function)

    fig = plt.figure()
    ax_1 = fig.add_subplot(2, 3, 1)
    ax_2 = fig.add_subplot(2, 3, 2)
    ax_3 = fig.add_subplot(2, 3, 3)
    ax_4 = fig.add_subplot(2, 3, 4)
    ax_5 = fig.add_subplot(2, 3, 5)

    ax_1.plot(arguments, function)
    ax_1.set(title='sin(3x) + cos(x)')

    ax_2.plot(arguments, hindrance_result)
    ax_2.set(title='sin(3x) + cos(x)(иск)')

    ax_3.plot(arguments, hff(hindrance_result, fc_hff))
    ax_3.set(title='однопол. ВЧ')
    ax_3.scatter(arguments, function, color='white')

    fft_res = abs(np.fft.rfft(hindrance_result))
    # fft_freq = np.fft.rfftfreq(N, 1/512)

    ax_4.plot(arguments, filter_band_Hamming(hindrance_result, fc, fc1))
    ax_4.set(title='Hamming')
    ax_4.scatter(arguments, function, color='white')
    ax_4.plot(arguments, function, color='orange')

    ax_5.plot(arguments2, fft_res)
    ax_5.set(title='FFT')
    ax_5.scatter(arguments, function, color='white')

    plt.show()
