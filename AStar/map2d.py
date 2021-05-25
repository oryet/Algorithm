# coding=utf-8
from __future__ import print_function
import numpy as np
import matplot.pltGrid as plt

class map2d:
    """
    地图数据
    """
    def __init__(self):
        '''
        self.data = [list("####################"),
                     list("#*****#************#"),
                     list("#*****#*****#*####*#"),
                     list("#*########*##******#"),
                     list("#*****#*****######*#"),
                     list("#*****#####*#******#"),
                     list("####**#*****#*######"),
                     list("#*****#**#**#**#***#"),
                     list("#**#*****#**#****#*#"),
                     list("####################")]
        '''
        self.data = [list("####################"),
                     list("#*****#************#"),
                     list("#*****#*****#*####*#"),
                     list("#*########*##******#"),
                     list("#*****#*****######*#"),
                     list("#*****#####*#******#"),
                     list("####**#*****#*######"),
                     list("#*****#**#**#**#***#"),
                     list("#**#**#**#**#****#*#"),
                     list("#*****#*****#******#"),
                     list("#*****#*****#*####*#"),
                     list("#*########*##******#"),
                     list("#*****#*****######*#"),
                     list("#*****#####*#******#"),
                     list("####**#*****#*######"),
                     list("#*****#**#**#**#***#"),
                     list("#**#*****#**#****#*#"),
                     list("####################")]

        self.w = 20
        self.h = 18
        self.passTag = '*'
        self.pathTag = 'o'
        self.map = np.full((self.h, self.w), int(0), dtype=np.int8)
        self.xList = []
        self.yList = []

    def showMap(self):
        for x in range(0, self.h):
            for y in range(0, self.w):
                print(self.data[x][y], end='')
                if self.data[x][y] == '*':
                    self.map[x][y] = 10
            print(" ")

        plt.showMap(self.map.T)
        return

    def setMap(self, point):
        self.data[point.x][point.y] = self.pathTag
        self.xList += [point.x]
        self.yList += [point.y]
        return

    def isPass(self, point):
        if (point.x < 0 or point.x > self.h - 1) or (point.y < 0 or point.y > self.w - 1):
            return False

        if self.data[point.x][point.y] == self.passTag:
            return True

    def showPath(self):
        plt.showPath(self.xList, self.yList)