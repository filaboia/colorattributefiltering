# This Python file uses the following encoding: utf-8
from __future__ import division
import numpy as np

class ImageAdapter(object):
    def adapt(self, f):
        raise NotImplementedError("Every ImageAdapter must implement the adapt method.")

class GrayImageAdapter(ImageAdapter):
    def adapt(self, f):
        return f,

class CoordinatesImageAdapter(ImageAdapter):
    def adapt(self, f):
        return f[0], f[1]

class ColorImageAdapter(ImageAdapter):
    def adapt(self, f):
        return f,

class ColorCoordinatesImageAdapter(ImageAdapter):
    def adapt(self, f):
        return f[:3].astype(np.uint8), f[3], f[4]