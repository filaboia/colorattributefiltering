# This Python file uses the following encoding: utf-8
from __future__ import division
import numpy as np

class ImageAdapter(object):
    def adapt(self, f):
        raise NotImplementedError("Every ImageAdapter must implement the adapt method.")

class CoordinatesImageAdapter(ImageAdapter):
    def adapt(self, f):
        return f[-2], f[-1]

class GrayImageAdapter(ImageAdapter):
    def adapt(self, f):
        if (len(f.shape) > 1):
            return f[0].astype(np.uint8),
        else:
            return f,

class GrayCoordinatesImageAdapter(ImageAdapter):
    def adapft(self, f):
        return f[0].astype(np.uint8), f[1], f[2]

class ColorImageAdapter(ImageAdapter):
    def adapt(self, f):
        return f[:3].astype(np.uint8),

class ColorCoordinatesImageAdapter(ImageAdapter):
    def adapt(self, f):
        return f[:3].astype(np.uint8), f[3], f[4]