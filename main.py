# This is a sample Python script.

import numpy as myNumpy # общие математические и числовые операции
import matplotlib.pyplot as plt # для построения графиков


class FastFurieTransfsorm:
    mulCounter = 0
    addCounter = 0
    @staticmethod # можно вызывать без создания экземпляра класса
    def fastFurieTransform(self, function, direction): # self чтобы ссылаться на самих себя
        self.mulCounter = 0
        self.addCounter = 0
        resultFunction = self.fastFurieTransformRecursive(self,function,direction)

        if direction == 1:
            valuesLenght = len(function)
            for i in range(valuesLenght):
                resultFunction[i] /= valuesLenght

        return resultFunction

    @staticmethod  # можно вызывать без создания экземпляра класса
    def fastFurieTransformRecursive(self, function, direction):
        valuesLenght = len(function)

        if valuesLenght == 1:
            return function
        firstHalf = [complex(0, 0)] * (int(valuesLenght/2)) # Подготовка массивов для операции "Бабочка"
        secondtHalf = [complex(0, 0)] * (int(valuesLenght / 2))

        Wn = complex(myNumpy.cos(2*myNumpy.pi/valuesLenght),direction * myNumpy.sin(2*myNumpy.pi/valuesLenght)) # Присвоить Wn значение главного комплексного корня N-й степени из единицы
        w = 1 # Присвоить w = 1
        result = [complex(0,0)]*valuesLenght
        for i in range(int(valuesLenght/2)): # Операция "Бабочка"
            firstHalf[i] = function[i] + function[i+int(valuesLenght/2)]
            secondtHalf[i] = (function[i] - function[i+int(valuesLenght/2)])*w
            w=w*Wn
            self.mulCounter += 1
            self.addCounter += 2

        evenResult = self.fastFurieTransformRecursive(self,firstHalf,direction) # Рекурсивный вызов БПФ для каждой из частей
        oddResult = self.fastFurieTransformRecursive(self,secondtHalf,direction)
        for i in range(valuesLenght): # Объединение результатов
            if i%2==0:
                result[i] = evenResult[int(i/2)]
            else:
                result[i] = oddResult[int(i/2)]

        return result

class CorrelationWithConvolution:
    mulCounter = 0
    addCounter = 0
    @staticmethod
    def correlationWithConvolutionOperation(firstFunction,secondFunction,operation):

        length = len(firstFunction)
        CorrelationWithConvolution.mulCounter = 0
        CorrelationWithConvolution.addCounter = 0

        # print("====")

        result = []
        for i in range(length):
            temp = 0
            for j in range(length):
                k = myNumpy.abs(i + (j * operation)) % length
                # k1 = (i + (j * operation)) % length
                # print(k)
                if (i + (j * operation)) < 0:
                    k = k * (-1)

                # print(k)
                temp += firstFunction[j] * secondFunction[k]

                CorrelationWithConvolution.mulCounter += 1
                CorrelationWithConvolution.addCounter += 1

            temp /= length
            result.append(temp)

        return result

    @staticmethod
    def FFTCorrelationWithConvolutionOperation(firstFunction, secondFuncton, operation):

        length = len(firstFunction)

        CorrelationWithConvolution.mulCounter = 0
        CorrelationWithConvolution.addCounter = 0

        firstFunctionFastFurieTransform = FastFurieTransfsorm.fastFurieTransform(FastFurieTransfsorm, firstFunction, 1)
        CorrelationWithConvolution.mulCounter += FastFurieTransfsorm.mulCounter
        CorrelationWithConvolution.addCounter += FastFurieTransfsorm.addCounter

        secondFunctionFastFurieTransform = FastFurieTransfsorm.fastFurieTransform(FastFurieTransfsorm, secondFuncton, 1)
        CorrelationWithConvolution.mulCounter += FastFurieTransfsorm.mulCounter
        CorrelationWithConvolution.addCounter += FastFurieTransfsorm.addCounter

        firstFunctionFastFurieTransformConjugate = firstFunctionFastFurieTransform
        if operation == 1:
            firstFunctionFastFurieTransformConjugate = myNumpy.conj(firstFunctionFastFurieTransform)

        temp = myNumpy.multiply(firstFunctionFastFurieTransformConjugate, secondFunctionFastFurieTransform)
        CorrelationWithConvolution.mulCounter += length

        result = FastFurieTransfsorm.fastFurieTransform(FastFurieTransfsorm, temp, -1)
        CorrelationWithConvolution.mulCounter += FastFurieTransfsorm.mulCounter
        CorrelationWithConvolution.addCounter += FastFurieTransfsorm.addCounter

        return result



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    N = 16
    arguments = myNumpy.arange(0, 16) * 2 * myNumpy.pi / N
    firstFunction = list(map(lambda x: myNumpy.cos(x), arguments))
    secondFunction = list(map(lambda x: myNumpy.sin(3 * x), arguments))
   # print('    : {}'.format(arguments))

    correlation = CorrelationWithConvolution.correlationWithConvolutionOperation(firstFunction, secondFunction, 1)
    print('Сложность корреляции (умножение): {}'.format(CorrelationWithConvolution.mulCounter))
    print('Сложность корреляции (сложение): {}'.format(CorrelationWithConvolution.addCounter))
    convolution = CorrelationWithConvolution.correlationWithConvolutionOperation(firstFunction, secondFunction, -1)
    print('Сложность свертки (умножение): {}'.format(CorrelationWithConvolution.mulCounter))
    print('Сложность свертки (сложение): {}'.format(CorrelationWithConvolution.addCounter))

    correlationWithFastFurieTransform = CorrelationWithConvolution.FFTCorrelationWithConvolutionOperation(firstFunction, secondFunction, 1)
    print('Сложность корреляции c БПФ (умножение): {}'.format(CorrelationWithConvolution.mulCounter))
    print('Сложность корреляции c БПФ (сложение): {}'.format(CorrelationWithConvolution.addCounter))
    convolutionWithFastFurieTransform = CorrelationWithConvolution.FFTCorrelationWithConvolutionOperation(firstFunction, secondFunction, -1)
    print('Сложность свертки c БПФ (умножение): {}'.format(CorrelationWithConvolution.mulCounter))
    print('Сложность свертки c БПФ (сложение): {}'.format(CorrelationWithConvolution.addCounter))

    fig = plt.figure()
    ax_1 = fig.add_subplot(3, 3, 1)
    ax_2 = fig.add_subplot(3, 3, 2)
    ax_3 = fig.add_subplot(3, 3, 3)
    ax_4 = fig.add_subplot(3, 3, 7)
    ax_5 = fig.add_subplot(3, 3, 8)
    ax_6 = fig.add_subplot(3, 3, 9)

    ax_1.plot(arguments, firstFunction)
    ax_1.set(title='cos(x)')
    ax_1.scatter(arguments, firstFunction, color='orange')
    ax_1.grid(True)

    ax_4.plot(arguments, secondFunction)
    ax_4.set(title='sin(3x)')
    ax_4.scatter(arguments, secondFunction, color='orange')
    ax_4.grid(True)

    ax_2.plot(arguments, correlation)
    ax_2.set(title='Корреляция')
    ax_2.scatter(arguments, correlation, color='orange')
    ax_2.grid(True)

    ax_3.plot(arguments, convolution)
    ax_3.set(title='Свертка')
    ax_3.scatter(arguments, convolution, color='orange')
    ax_3.grid(True)

    ax_5.plot(arguments, correlationWithFastFurieTransform)
    ax_5.set(title='Корреляция с БПФ')
    ax_5.scatter(arguments, correlationWithFastFurieTransform, color='orange')
    ax_5.grid(True)

    ax_6.plot(arguments, convolutionWithFastFurieTransform)
    ax_6.set(title='Свертка с БПФ')
    ax_6.scatter(arguments, convolutionWithFastFurieTransform, color='orange')
    ax_6.grid(True)

    plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
