# This Python file uses the following encoding: utf-8
from __future__ import division
from colorsegmentation.image.ImageService import *
from colorsegmentation.image.ImageValidator import *
from colorsegmentation.image.ImageAdapter import *
import numpy as np

class ImageFunction(object):
    def __new__(cls, f):
        f = np.array(f)
        cls.validate(f)
        return cls.compute(*cls.adapt(f))
    
    @staticmethod
    def validate(f):
        raise NotImplementedError("Every ImageFunction must implement the validate static method.")
    
    @staticmethod
    def compute(f):
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
    def compute(f):
        count = np.bincount(normaliza(f+1))[1:]
        p = count/np.sum(count)
        log = np.log2(p);
        return abs(np.sum(p*log))

class Area(GrayFunction):
    @staticmethod
    def compute(f):
        return np.sum(f >= 0)

class Height(GrayFunction):
    @staticmethod
    def compute(f):
        return np.max(f) - np.min(f)

class Volume(GrayFunction):
    @staticmethod
    def compute(f):
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
    def compute(f):
        return pow(np.sum(pow(np.sum(pow(f - np.mean(f, axis=1,  keepdims=True), 2), axis=0), 0.5)), 2)

class ColorHarmony(ColorFunction):
    @staticmethod
    def compute(f):
        def two_colors_harmony(p1, p2):
            def atualiza_range(p):
                novop = np.empty(p.shape, dtype=np.float64)    
                novop[0] = p[0] / 255. * 100. 
                novop[1] = p[1] / 255. * 254. - 127.
                novop[2] = p[2] / 255. * 254. - 127.
                return novop
            
            def cab(p):
                return pow(pow(p[1], 2) + pow(p[2], 2), 0.5)
            
            def hab(p):
                return np.arctan2(p[2], p[1]) % (2 * np.pi)
            
            p1 = atualiza_range(p1)
            p2 = atualiza_range(p2)
            
            deltaa = abs(p1[1] - p2[1])
            deltab = abs(p1[2] - p2[2])
            deltaCab = cab(p1) - cab(p2)
            deltahab = ((hab(p1) - hab(p2) + np.pi) % (2 * np.pi)) - np.pi
            deltaHab = 2 * pow(cab(p1) * cab(p2), 0.5) * np.sin(deltahab/2.) 
            deltaC = pow(pow(deltaHab, 2) + pow(deltaCab / 1.46, 2), 0.5)
            HC = 0.04 + 0.53 * np.tanh(0.8 - 0.045 * deltaC)
            
            Lsum = p1[0] + p2[0]
            HLsum = 0.28 + 0.54 * np.tanh(-3.88 + 0.029 * Lsum)
            deltaL = abs(p1[0] - p2[0])
            HdeltaL = 0.14 + 0.15 * np.tanh(-2 + 0.2*deltaL)
            HL = HLsum + HdeltaL
            
            def ec(p):
                return 0.5 + 0.5 * np.tanh(-2 + 0.5 * cab(p))
            
            def hs(p):
                return -0.08 - 0.14 * np.sin(hab(p) + np.pi/3.6) - 0.07 * np.sin(2 * hab(p) + np.pi/2)
            
            def ey(p):
                return ((0.22*p[0] - 12.8) / 10) * np.exp((90 - np.degrees(hab(p))) / 10 - np.exp((90 - np.degrees(hab(p))) / 10))
            
            def hsy(p):
                return ec(p) * (hs(p) + ey(p))
            
            HH = hsy(p1) + hsy(p2)
            
            return HC + HL + HH
        
        MxN = f.shape[1]
        
        return np.sum([np.sum(two_colors_harmony(np.full_like(f[:, i+1:], f[:, i:i+1]), f[:, i+1:])) for i in range(MxN - 1)]) / ((MxN - 1) * MxN / 2)

class ColorCoordinatesFunction(ImageFunction):
    @staticmethod    
    def validate(f):
        ColorCoordinatesImageValidator().validate(f)
    
    @staticmethod    
    def adapt(f):
        return ColorCoordinatesImageAdapter().adapt(f)