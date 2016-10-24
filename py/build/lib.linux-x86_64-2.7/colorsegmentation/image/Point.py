# This Python file uses the following encoding: utf-8
from __future__ import division
import numpy as np

def points(f, dtype):
    points = np.nonzero(f)
    return (points[0].astype(dtype), points[1].astype(dtype))    
    
def filamergepoints(a, b):
    return (np.concatenate((a[0], b[0])), np.concatenate((a[1], b[1])))
    
def filapointstoimage(points):
    offset = (points[0].min(), points[1].min())
    image = np.zeros((points[0].max() - offset[0] + 1, points[1].max() - offset[1] + 1), np.bool)
    image[points[0] - offset[0], points[1] - offset[1]] = 1
    return image
