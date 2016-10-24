# This Python file uses the following encoding: utf-8
from __future__ import division
import numpy as np

class Points(object):
    def __init__(self, dtype, f=None):
        if (f is None):
            self.coordX = np.array([], dtype) 
            self.coordY = np.array([], dtype)
        else:
            self.coordX, self.coordY = np.nonzero(f)
            self.coordX = self.coordX.astype(dtype)
            self.coordY = self.coordY.astype(dtype)
    
    def merge(self, points):
        self.coordX = np.concatenate((self.coordX, points.getX()))
        self.coordY = np.concatenate((self.coordY, points.getY()))
        return self()
    
    def toImage(self):
        offsetX = self.coordX.min()
        offsetY = self.coordY.min()
        image = np.zeros((self.coordX.max() - offsetX + 1, self.coordY.max() - offsetY + 1), np.bool)
        image[self.coordX - offsetX, self.coordY - offsetY] = 1
        return image
    
    def getX(self):
        return self.coordX
    
    def setX(self, coordX):
        self.coordX = coordX
    
    def getY(self):
        return self.coordY
    
    def setY(self, coordY):
        self.coordY = coordY
    
    def __call__(self):
        return (self.coordX, self.coordY)