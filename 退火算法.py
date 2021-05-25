from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math


START = 0
END = 100
DLEN = END - START

# define aim function
def aimFunction(x):
    # y = x ** 3 - 60 * x ** 2 - 4 * x + 6
    # y = 3 * x ** 3 - 5 * x ** 2 - 6 * x + 7
    y = x**2 + 2*x - 2
    # y = math.exp(x)

    return y

def showFun(x1, y1, ):
    x = [i/10 for i in range(START, END)]
    y = [0 for i in range(DLEN)]
    for i in range(DLEN):
        y[i] = aimFunction(x[i])

    plt.plot(x, y)
    plt.plot(x1, y1, 'o')
    plt.show()


def showDynFun(x1, y1, ):
    x = [i/10 for i in range(START, END)]
    y = [0 for i in range(DLEN)]
    for i in range(DLEN):
        y[i] = aimFunction(x[i])

    # 打开交互模式
    plt.ion()

    plt.plot(x, y)

    for i in range(len(x1)):
        # 暂停
        plt.pause(0.01)
        plt.plot(x1[i], y1[i], 'o')

    # 关闭交互模式
    plt.ioff()
    plt.show()


def tuiHuo():
    T = DLEN  # initiate temperature
    Tmin = 10  # minimum value of terperature
    x = np.random.uniform(low=START, high=END)  # initiate x
    k = 50  # times of internal circulation
    y = 0  # initiate result
    t = 0  # time
    ylist = []
    xlist = []
    while T >= Tmin:
        for i in range(k):
            # calculate y
            y = aimFunction(x)
            # generate a new x in the neighboorhood of x by transform function
            xNew = x + np.random.uniform(low=-0.055, high=0.055) * T
            if (START <= xNew and xNew <= END):
                yNew = aimFunction(xNew)
                if yNew - y < 0:
                    x = xNew
                    xlist += [x]
                    ylist += [yNew]
                else:
                    # metropolis principle
                    p = math.exp(-(yNew - y) / T)
                    r = np.random.uniform(low=0, high=1)
                    if r < p:
                        x = xNew
        t += 1
        print(t)
        T = DLEN / (1 + t)

    y = aimFunction(x)
    print(x, y)
    return (xlist, ylist)

if __name__ == '__main__':
    x, y = tuiHuo()
    showDynFun(x, y)
