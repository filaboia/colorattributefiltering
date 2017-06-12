# This Python file uses the following encoding: utf-8
from __future__ import division
from colorsegmentation.image.ImageService import *
from colorsegmentation.image.ImageValidator import *
from colorsegmentation.image.ImageAdapter import *
from colorsegmentation.image.StatisticsService import *
import numpy as np
import numpy.ma as ma

class ImageFunction(object):
    def __new__(cls, f):
        f = np.array(f)
        cls.validate(f)
        return cls.compute(cls, *cls.adapt(f))
    
    @staticmethod
    def validate(f):
        raise NotImplementedError("Every ImageFunction must implement the validate static method.")
    
    @staticmethod
    def compute(cls, f):
        raise NotImplementedError("Every ImageFunction must implement the compute static method.")
    
    @staticmethod
    def adapt(f):
        raise NotImplementedError("Every ImageFunction must implement the adapt static method.")

class GrayFunction(ImageFunction):
    @staticmethod
    def validate(f):
        GrayImageValidator().validate(f)
    
    @staticmethod
    def adapt(f):
        return GrayImageAdapter().adapt(f)

class Entropy(GrayFunction):
    @staticmethod
    def compute(cls, f):
        count = np.bincount(normaliza(f+1))[1:]
        p = count/np.sum(count)
        log = np.log2(p);
        return abs(np.sum(p*log))

class Area(GrayFunction):
    @staticmethod
    def compute(cls, f):
        return np.sum(f >= 0)

class Height(GrayFunction):
    @staticmethod
    def compute(cls, f):
        return np.max(f) - np.min(f)

class Volume(GrayFunction):
    @staticmethod
    def compute(cls, f):
        return np.sum(f - np.min(f) + 1)

class CoordinatesFunction(ImageFunction):
    @staticmethod
    def validate(f):
        CoordinatesImageValidator().validate(f)
    
    @staticmethod
    def adapt(f):
        return CoordinatesImageAdapter().adapt(f)

class ColorFunction(ImageFunction):
    @staticmethod
    def validate(f):
        ColorImageValidator().validate(f)
    
    @staticmethod
    def adapt(f):
        return ColorImageAdapter().adapt(f)

class AverageColorError(ColorFunction):
    @staticmethod
    def compute(cls, f):
        return np.sum(pow(np.sum(pow(f - np.mean(f, axis=1,  keepdims=True), 2), axis=0), 0.5))

class HistogramDivergenceFunction(ColorFunction):
    @staticmethod
    def compute(cls, f):
        h = weightedHueHistogram(f)
        return kullbackLieblerDivergence(h, cls.baseHistogram())
        
    @staticmethod
    def baseHistogram():
        raise NotImplementedError("Every HistogramDivergenceFunction must implement the adapt static method.")

class ColorCoordinatesFunction(ImageFunction):
    @staticmethod    
    def validate(f):
        ColorCoordinatesImageValidator().validate(f)
    
    @staticmethod    
    def adapt(f):
        return ColorCoordinatesImageAdapter().adapt(f)

class ColorHarmony(ColorCoordinatesFunction):
    @staticmethod
    def compute(cls, f, x, y):
        g = ma.empty((3, x.max() - x.min() + 1, y.max() - y.min() + 1), dtype=f.dtype)
        g.mask = True
        
        g[:, x - x.min(), y - y.min()] = f[:]
        
        return float(ma.mean(gradient(g, gradientType=3, distanceType='harmony', includeOrigin=True, normalize=False)))