# This Python file uses the following encoding: utf-8
from __future__ import division
from colorsegmentation.image.ImageService import *
import numpy as np

class ImageValidator(object):
    def validate(self, f):
        raise NotImplementedError("Every ImageValidator must implement the validate method.")

class CoordinatesImageValidator(ImageValidator):
    def validate(self, f):
        try: 
            ColorCoordinatesImageValidator().validate(f)
        except TypeError:
            try:
                GrayCoordinatesImageValidator().validate(f)
            except TypeError:
                if (np.size(f.shape) != 2 or f.shape[0] != 2):
                    raise TypeError("Input should be in the xy form but had shape " + str(f.shape) + ".")
        
class GrayImageValidator(ImageValidator):
    def validate(self, f):
        try:
            GrayCoordinatesImageValidator().validate(f)
        except TypeError:
            if (np.size(f.shape) > 1):
                raise TypeError("Input should be in the gray f form but had shape " + str(f.shape) + ".")

class GrayCoordinatesImageValidator(ImageValidator):
    def validate(self, f):
        if (np.size(f.shape) != 2 or f.shape[0] != 3):
            raise TypeError("Input should be in the color fxy form but had shape " + str(f.shape) + ".")

class ColorImageValidator(ImageValidator):
    def validate(self, f):
        try: 
            ColorCoordinatesImageValidator().validate(f)
        except TypeError:
            if (np.size(f.shape) != 2 or f.shape[0] != 3):
                raise TypeError("Input should be in the color f form but had shape " + str(f.shape) + ".")

class ColorCoordinatesImageValidator(ImageValidator):
    def validate(self, f):
        if (np.size(f.shape) != 2 or f.shape[0] != 5):
            raise TypeError("Input should be in the fxy form but had shape " + str(f.shape) + ".")