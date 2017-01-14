# This Python file uses the following encoding: utf-8
from __future__ import division
from colorsegmentation.image.ImageService import *
from colorsegmentation.image.ImageValidator import *
from colorsegmentation.image.ImageAdapter import *
import numpy as np

class ImageOperator(object):
    def __new__(cls, f1, f2):
        cls.validate(f1)
        cls.validate(f2)
        return cls.compute(*cls.adapt(f1) + cls.adapt(f2))
    
    @staticmethod
    def validate(f):
        raise NotImplementedError("Every ImageOperator must implement the validate static method.")
    
    @staticmethod
    def compute(f1, f2):
        raise NotImplementedError("Every ImageOperator must implement the compute static method.")
    
    @staticmethod
    def adapt(f):
        raise NotImplementedError("Every ImageOperator must implement the adapt static method.")

class ColorCoordinatesImageOperator(ImageOperator):
    @staticmethod
    def validate(f):
        ColorCoordinatesImageValidator().validate(f)
    
    @staticmethod
    def adapt(f):
        return ColorCoordinatesImageAdapter().adapt(f)

# class HarmoniaCorOperator(ColorCoordinatesImageOperator):
#     @staticmethod
#     def compute(f1, x1, y1, f2, x2, y2):