# This Python file uses the following encoding: utf-8
from __future__ import division
from colorsegmentation.image.ImageService import *
from colorsegmentation.image.ImageFunction import *
import numpy as np

class ImageEvaluationFunction(object):
    def __new__(cls, f, w):
        if (np.prod(f.shape[-2:]) != np.prod(w.shape[-2:])):
            raise TypeError("Inputs' last two dimensions must match but had shape " + str(f.shape) + " and " + str(w.shape) + ".")
        
        return cls.compute(f, w)
    
    @staticmethod
    def compute(f, w):
        raise NotImplementedError("Every EvaluationFunction must implement the compute static method.")

class AverageColorErrorDerived(ImageEvaluationFunction):
    @staticmethod
    def compute(f, w):
        return grain(w, f, AverageColorError, 'data', True).squeeze(axis=1)

class AverageColorErrorWeighted(ImageEvaluationFunction):
    @staticmethod
    def compute(f, w):
        return np.mean(grain(w, f, AverageColorError, 'image', True))

class AverageColorErrorVariance(ImageEvaluationFunction):
    @staticmethod
    def compute(f, w):
        return np.var(grain(w, f, AverageColorError, 'data', True))

class LiuF(AverageColorErrorDerived):
    @staticmethod
    def compute(f, w):
        w = normaliza(w)
        A = region_area(w)
        R = w.max()
        e2 = pow(AverageColorErrorDerived(f, w), 2)
        return pow(R, 0.5) * np.sum(e2 / pow(A, 0.5))

class BorsottiF(AverageColorErrorDerived):
    @staticmethod
    def compute(f, w):
        w = normaliza(w)
        A = region_area(w)
        e2 = pow(AverageColorErrorDerived(f, w), 2)
        R = w.max()
        NxM = np.sum(A)
        return pow(R, 0.5) / (10000 * NxM) * pow(np.sum(pow(np.bincount(A)[1:], 1 + 1 / (np.arange(A.max()) + 1))) , .5) * np.sum(e2 / pow(A, 0.5))

class BorsottiQ(AverageColorErrorDerived):
    @staticmethod
    def compute(f, w):
        w = normaliza(w)
        A = region_area(w)
        e2 = pow(AverageColorErrorDerived(f, w), 2)
        R = w.max()
        NxM = np.sum(A)
        return pow(R, 0.5) / (10000 * NxM) * np.sum(e2 / (1 + np.log10(A)) + pow(count_regiao(A) / A, 2))

class EntropyDerived(ImageEvaluationFunction):
    @staticmethod
    def compute(f, w):
        return grain(w, normaliza(codifica(f)), Entropy, 'data').squeeze(axis=1)

class EntropyWeighted(ImageEvaluationFunction):
    @staticmethod
    def compute(f, w):
        return np.mean(grain(w, normaliza(codifica(f)), Entropy))

class EntropyVariance(ImageEvaluationFunction):
    @staticmethod
    def compute(f, w):
        return np.var(grain(w, normaliza(codifica(f)), Entropy, 'data'))

class EntropySum(ImageEvaluationFunction):
    @staticmethod
    def compute(f, w):
        return np.sum(grain(w, normaliza(codifica(f)), Entropy))

class ZhangHw(EntropyWeighted):
    @staticmethod
    def compute(f, w):
        w = normaliza(w)
        R = w.max()
        return pow(R, 0.5) * EntropyWeighted(f, w)

class ZhangE(EntropyWeighted):
    @staticmethod
    def compute(f, w):
        return EntropyWeighted(f, w) + Entropy(w.ravel())

class ColorHarmonyDerived(ImageEvaluationFunction):
    @staticmethod
    def compute(f, w):
        return grain(w.ravel()[...,np.newaxis], fxyform(f)[...,np.newaxis], ColorHarmony, 'data', True).squeeze(axis=1)

class ColorHarmonyWeighted(ImageEvaluationFunction):
    @staticmethod
    def compute(f, w):
        return np.mean(grain(w.ravel()[...,np.newaxis], colorform(f)[...,np.newaxis], ColorHarmony, 'image', True))

class ColorHarmonySegmented(ColorHarmonyWeighted):
    @staticmethod
    def compute(f, w):
        return ColorHarmonyWeighted(grain(w, f, 'mean').astype(f.dtype), (w >= 0).astype(w.dtype))
