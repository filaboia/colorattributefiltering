# This Python file uses the following encoding: utf-8
from __future__ import division
from colorsegmentation.image.ImageService import *
import numpy as np

class ImageFunction(object):
    def __new__(cls):
        return cls.compute()
    
    @staticmethod
    def compute():
        raise NotImplementedError("Every ImageFunction must implement the compute static method.")

class GrayFunction(ImageFunction):
    def __new__(cls, f):
        if (np.size(f.shape) != 1):
            raise TypeError("Input should be in the gray f form but had shape " + str(f.shape) + ".")
        return cls.compute(f)
    
    @staticmethod
    def compute(f):
        raise NotImplementedError("Every GrayFunction must implement the compute static method.")

class Entropia(GrayFunction):
    @staticmethod
    def compute(f):
        count = np.bincount(normaliza(f))
        p = count/np.sum(count)
        log = np.log2(p); log[np.isinf(log)] = 0
        return -np.sum(p*log)

class StructureFunction(ImageFunction):
    def __new__(cls, xy):
        if (np.size(xy.shape) != 2 or xy.shape[0] != 2):
            raise TypeError("Input should be in the xy form but had shape " + str(xy.shape) + ".")
        
        return cls.compute(xy[0], xy[1])
        
        @staticmethod
        def compute(x, y):
            raise NotImplementedError("Every StructureFunction must implement the compute static method.")

class Area(StructureFunction):
    @staticmethod
    def compute(x, y):
        return x.shape[0]

class ColorFunction(ImageFunction):
    def __new__(cls, f):
        if (np.size(f.shape) != 2 or f.shape[0] != 3):
            raise TypeError("Input should be in the color f form but had shape " + str(f.shape) + ".")
        
        return cls.compute(f)
    
    @staticmethod
    def compute(f):
        raise NotImplementedError("Every ColorFunction must implement the compute static method.")

class ErroMedioQuadratico(ColorFunction):
    @staticmethod
    def compute(f):
        return np.sum(pow(f - np.mean(f, axis=1,  keepdims=True), 2))

class ColorStructureFunction(ImageFunction):
    def __new__(cls, fxy):
        if (np.size(fxy.shape) != 2 or fxy.shape[0] != 5):
            raise TypeError("Input should be in the fxy form but had shape " + str(fxy.shape) + ".")
        
        return cls.compute(fxy[:3].astype(np.uint8), fxy[3], fxy[4])
    
        @staticmethod
        def compute(f, x, y):
            raise NotImplementedError("Every ColorStructureFunction must implement the compute static method.")

class HarmoniaCor(ColorStructureFunction):
    @staticmethod
    def compute(f, x, y):
        def harmonia_duas_cores(p1, p2):
            def cab(p):
                return pow(pow(p[1], 2) + pow(p[2], 2), 0.5)
            
            def hab(p):
                return np.arctan2(p[2], p[1]) % (2 * np.pi)
            
            def Hlinha(hab):
                return -0.23 - 0.35 * np.sin(hab + 0.83) - 0.18 * np.sin(2 * hab + 1.55)
            
            def atualiza_range(p):
                novop = np.empty(p.shape, dtype=np.float16)    
                novop[0] = p[0] / 255. * 100. 
                novop[1] = p[1] / 255. * 254. - 127.
                novop[2] = p[2] / 255. * 254. - 127.
                return novop
            
            p1 = atualiza_range(p1)
            p2 = atualiza_range(p2)
            Lsum = p1[0] + p2[0]
            deltaCab = abs(cab(p1) - cab(p2))
            deltaL = abs(p1[0] - p2[0])
            deltaa = abs(p1[1] - p2[1])
            deltab = abs(p1[2] - p2[2])
            deltaHab = np.nan_to_num(pow(pow(deltaa, 2) + pow(deltab, 2) - pow(deltaCab, 2), 0.5))
            deltaC = pow(deltaHab + pow(deltaCab / 1.5, 2), 0.5)
            HdeltaC = 1.3 - 0.07 * deltaC + 0.0005 * pow(deltaC, 2)
            HdeltaL = -0.92 + 0.05 * deltaL - 0.0006 * pow(deltaL, 2)
            Hh = Hlinha(hab(p1)) + Hlinha(hab(p2))
            return -2.2 + 0.03 * Lsum + HdeltaC + HdeltaL + 1.1 * Hh
        
        MxN = f.shape[1]
        
        return np.sum([np.sum(harmonia_duas_cores(np.full_like(f[:, i+1:], f[:, i:i+1]), f[:, i+1:]) 
        / pow(pow(np.full_like(x[i+1:], x[i:i+1]) - x[i+1:], 2) + pow(np.full_like(y[i+1:], y[i:i+1]) - y[i+1:], 2), 0.5)) 
        for i in range(MxN - 1)]) / (2 * MxN)

class EvaluationFunction(ImageFunction):
    def __new__(cls, f, w):
        if (np.prod(f.shape[-2:]) != np.prod(w.shape[-2:])):
            raise TypeError("Inputs' last two dimensions must match but had shape " + str(f.shape) + " and " + str(w.shape) + ".")
        
        return cls.compute(f, w)
    
    @staticmethod
    def compute(f, w):
        raise NotImplementedError("Every EvaluationFunction must implement the compute static method.")

class ErroMedioQuadraticoDerived(EvaluationFunction):
    @staticmethod
    def compute(f, w):
        return filagrain(w, f, ErroMedioQuadratico, 'data', True).squeeze(axis=1)

class ErroMedioQuadraticoPonderado(EvaluationFunction):
    @staticmethod
    def compute(f, w):
        return np.mean(filagrain(w, f, ErroMedioQuadratico, 'image', True))

class ErroMedioQuadraticoVariancia(EvaluationFunction):
    @staticmethod
    def compute(f, w):
        return np.var(filagrain(w, f, ErroMedioQuadratico, 'data', True))

class LiuF(ErroMedioQuadraticoDerived):
    @staticmethod
    def compute(f, w):
        w = normaliza(w)
        A = region_area(w)
        R = w.max()
        e2 = ErroMedioQuadraticoDerived(f, w)
        return pow(R, 0.5) * np.sum(e2 / pow(A, 0.5))

class BorsottiF(ErroMedioQuadraticoDerived):
    @staticmethod
    def compute(f, w):
        w = normaliza(w)
        A = region_area(w)
        e2 = ErroMedioQuadraticoDerived(f, w)
        R = w.max()
        NxM = np.sum(A)
        return pow(R, 0.5) / (10000 * NxM) * pow(np.sum(pow(np.bincount(A)[1:], 1 + 1 / (np.arange(A.max()) + 1))) , .5) * np.sum(e2 / pow(A, 0.5))

class BorsottiQ(ErroMedioQuadraticoDerived):
    @staticmethod
    def compute(f, w):
        w = normaliza(w)
        A = region_area(w)
        e2 = ErroMedioQuadraticoDerived(f, w)
        R = w.max()
        NxM = np.sum(A)
        return pow(R, 0.5) / (10000 * NxM) * np.sum(e2 / (1 + np.log10(A)) + pow(count_regiao(A) / A, 2))

class EntropiaDerived(EvaluationFunction):
    @staticmethod
    def compute(f, w):
        return filagrain(w, normaliza(codifica(f)), Entropia, 'data').squeeze(axis=1)

class EntropiaPonderada(EvaluationFunction):
    @staticmethod
    def compute(f, w):
        return np.mean(filagrain(w, normaliza(codifica(f)), Entropia))

class EntropiaVariancia(EvaluationFunction):
    @staticmethod
    def compute(f, w):
        return np.var(filagrain(w, normaliza(codifica(f)), Entropia, 'data'))

class EntropiaSoma(EvaluationFunction):
    @staticmethod
    def compute(f, w):
        return np.sum(filagrain(w, normaliza(codifica(f)), Entropia))

class DesordemZhang(EntropiaPonderada):
    @staticmethod
    def compute(f, w):
        w = normaliza(w)
        R = w.max()
        return pow(R, 0.5) * EntropiaPonderada(f, w)

class EntropiaZhang(EntropiaPonderada):
    @staticmethod
    def compute(f, w):
        return EntropiaPonderada(f, w) + Entropia(w.ravel())

class HarmoniaCorDerived(EvaluationFunction):
    @staticmethod
    def compute(f, w):
        return filagrain(w.ravel()[...,np.newaxis], fxyform(f)[...,np.newaxis], HarmoniaCor, 'data', True).squeeze(axis=1)

class HarmoniaCorPonderada(EvaluationFunction):
    @staticmethod
    def compute(f, w):
        return np.mean(filagrain(w.ravel()[...,np.newaxis], fxyform(f)[...,np.newaxis], HarmoniaCor, 'image', True))

class HarmoniaCorSegmentada(HarmoniaCorPonderada):
    @staticmethod
    def compute(f, w):
        return HarmoniaCorPonderada(filagrain(w, f, 'mean').astype(f.dtype), (w >= 0).astype(w.dtype))
