import numpy as np  # общие математические и числовые операции
import matplotlib.pyplot as plt  # для построения графиков



def filter_band_Hamming(function,fc,fc1):
    M = len(function)
    h = [0] * M
    for i in range(M):
        if((i-M/2)==0):
            h[i]=2*np.pi*fc
        else:
            h[i] = (np.sin(2*np.pi*fc*(i-M/2)))/(i-M/2)*(0.54 - 0.46 * np.cos((2 * np.pi * i) / M))



    h1 = [0] * M
    for i in range(M):
        if ((i - M / 2) == 0):
            h1[i] = 2 * np.pi * fc1
        else:
            h1[i] = (np.sin(2 * np.pi * fc1 * (i - M / 2))) / (i - M / 2) * (0.54 - 0.46 * np.cos((2 * np.pi * i) / M))

    sum = 0
    for i in range(M):
        sum += h[i]
    for i in range(M):
        h[i] /= sum

    sum1 = 0
    for i in range(M):
        sum1 += h1[i]
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

    res = []
    # j=M
    for i in range(M):
        temp =0
        for j in range(M):
            k = np.abs(i-j)%M
            temp += function[k]*h2[j]

        temp/=M
        res.append(temp)
    return res


def add_hindrance(function):
    n = len(function)
    result = [0] * n

    for i in range(n):
        # result[i] = (function[i]) + np.cos(35 * 2 * np.pi * i / N)
        if(i > int(n / 3)):
            if(i < int(n / 3 + 0.2*n)):
                result[i] = (function[i]) + np.cos(75 * 2 * np.pi * i / N)
            else:
                result[i] = (function[i])
        else:
            result[i] = (function[i])
    return result



def hff(func, fc):
    x = np.exp((-2)*np.pi*fc)
    a0 = (1 + x) / 2
    a1 = (-1 - x) / 2
    b1 = x

    N = len(func)
    y = [0] * N
    for i in range(N):
        if(i>1):
            y[i] = a0 * func[i] + a1 * func[i-1] + b1 * y[i-1]

    return y


if __name__ == '__main__':
    N = 128
    fc =0.34
    fc1 =0.4
    arguments = np.arange(0, N) * 2 * np.pi / N
    function = list(map(lambda x: np.sin(3 * x) + np.cos(1*x), arguments))

    hindrance_result = add_hindrance(function)

    fig = plt.figure()
    ax_1 = fig.add_subplot(2, 2, 1)
    ax_2 = fig.add_subplot(2, 2, 2)
    ax_3 = fig.add_subplot(2, 2, 3)
    ax_4 = fig.add_subplot(2, 2, 4)

    ax_1.plot(arguments, function)
    ax_1.set(title='sin(3x) + cos(x)')

    ax_2.plot(arguments, hindrance_result)
    ax_2.set(title='sin(3x) + cos(x)(иск)')

    ax_3.plot(arguments, hff(hindrance_result, fc))
    ax_3.set(title='однопол. ВЧ')
    ax_3.scatter(arguments, function, color='white')

    ax_4.plot(arguments, filter_band_Hamming(hindrance_result,fc,fc1))
    ax_4.set(title='Hamming')

    plt.show()
