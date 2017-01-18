# This Python file uses the following encoding: utf-8
from __future__ import division
import numpy as np
import pandas as pd
from morph import *
            
def gradient(f, gradientType=1, distanceType='euclid'):
    def desloca(f, dx, dy):
        g = np.copy(f)
        zmax, xmax, ymax = f.shape
        g[:,np.clip(dx, 0, xmax):np.clip(xmax+dx, 0, xmax),np.clip(dy, 0, ymax):np.clip(ymax+dy, 0, ymax)] = f[:,np.clip(-dx, 0, xmax):np.clip(xmax-dx, 0, xmax),np.clip(-dy, 0, ymax):np.clip(ymax-dy, 0, ymax)]
        return g
    
    def euclid(f, fd):
        return pow(np.sum(pow(f - fd, 2), axis=0),0.5)
    
    def taxi(f, fd):
        return np.sum(abs(f - fd), axis=0)
    
    def chess(f, fd):
        return np.max(abs(f - fd), axis=0)
    
    def identity(f, fd=None):
        return f
    
    try:    
        distance = locals()[distanceType]
    except:
        distance = identity
    
    fdtype = f.dtype
    max_dtype = np.iinfo(fdtype).max
    
    f = f.astype(np.float)
    
    distances = [distance(f, desloca(f, 0, 1)), distance(f, desloca(f, 0, -1)), distance(f, desloca(f, 1, 0)), distance(f, desloca(f, -1, 0))]
    
    if gradientType == 1:
        g = np.max(distances, axis=0) 
    elif gradientType == 2:
        g = np.max(distances, axis=0) - np.min(distances, axis=0) 
    elif gradientType == 3:
        distances += [distance(desloca(f, 0, 1), desloca(f, 0, -1)), distance(desloca(f, 0, 1), desloca(f, 1, 0)), distance(desloca(f, 0, 1), desloca(f, -1, 0))]
        distances += [distance(desloca(f, 0, -1), desloca(f, 1, 0)), distance(desloca(f, 0, -1), desloca(f, -1, 0))]
        distances += [distance(desloca(f, 1, 0), desloca(f, -1, 0))]
        g = np.max(distances, axis=0) 
    else:
        raise ValueError
    
    g = (g - g.min()) / (g.max() - g.min()) * max_dtype   
        
    return np.clip(np.round(g), 0, max_dtype).astype(fdtype)    
    
def grain(fr, f, function, option='image', combineBands=False):
    def codifica(f):
        f = f.astype(np.uint64)
        cf = np.zeros(f.shape[1], dtype=np.uint64)
        offset = 64
        offset -= 8
        cf |= f[0] << offset
        if (f.shape[0] > 1):
            offset -= 8
            cf |= f[1] << offset
            offset -= 8
            cf |= f[2] << offset
        if (f.shape[0] > 3):
            offset -= 16
            cf |= f[3] << offset
            offset -= 16
            cf |= f[4] << offset
        return cf

    def decodifica(cf):
        cf = cf.astype(np.uint64)
        offset = 64
        offset -= 8
        f0 = (cf // 2 ** offset).astype(np.uint8)
        offset -= 8
        f1 = (cf // 2 ** offset).astype(np.uint8)
        offset -= 8
        f2 = (cf // 2 ** offset).astype(np.uint8)
        offset -= 16
        f3 = (cf // 2 ** offset).astype(np.uint16)
        offset -= 16
        f4 = (cf // 2 ** offset).astype(np.uint16)
        if (np.any(f3) or np.any(f4)):
            return np.array([f0, f1, f2, f3, f4], dtype=np.uint16)
        if (np.any(f1) or np.any(f2)):
            return np.array([f0, f1, f2], dtype=np.uint8)
        return f0

    def aplica(cf):
        return function(decodifica(cf))
    
    hasbackground = np.bincount(fr.ravel())[0] > 0
    if len(f.shape) == 1:
        f = f[np.newaxis]

    bidimensional = len(f.shape) == 2
    if bidimensional:
        f = f[np.newaxis]

    prep = f.reshape(f.shape[0], f.shape[1] * f.shape[2])

    if (combineBands):
        prep = codifica(prep)

    prep = np.vstack((fr.ravel(), prep)).T
    
    dt = pd.DataFrame(prep)
    labelled_image = dt.groupby(0)

    if (combineBands):
        LUT = labelled_image.agg(aplica).values
    else:
        LUT = labelled_image.agg(function).values

    if hasbackground:
        LUT = LUT[1:]

    indices = np.bincount(np.unique(fr))[1:]
    indices = np.cumsum(indices) * indices
    LUT = np.vstack((np.zeros(LUT.shape[1], dtype=LUT.dtype), LUT))[indices]
    
    if option == 'data':
        return LUT
        
    LUT = np.vstack((np.zeros(LUT.shape[1], dtype=LUT.dtype), LUT))
        
    LUT = np.squeeze(LUT)
        
    image = LUT[fr]
    if not bidimensional and not combineBands:
        image = np.rollaxis(image, 2)
    
    return image
    
def ordena_minimos(mins, inversa=False):
    args = np.argsort(mins.ravel())
    if inversa:
        args = args[::-1]
    o_mins = np.argsort(args).reshape(mins.shape) + 1
    reg_mins = grain(mmlabel(~mins.mask), o_mins, 'min')
    count = np.bincount(np.unique(reg_mins))
    count[0] = 0
    return np.cumsum(count)[reg_mins]

def normaliza(w):
    return (np.cumsum(np.concatenate(([0], np.bincount(w.ravel())[1:])) > 0)[w]).astype(w.dtype)

def count_regiao(A):
    return np.bincount(A)[A]

def region_area(w):
    return np.bincount(w.ravel())[1:] 

def codifica(f):
    f = f.astype(np.uint32)
    return f[0] * 65536 + f[1] * 256 + f[2]

def decodifica(f):
    f0 = f / 65536
    faux = f - (f0 * 65536)
    f1 = faux / 256
    f2 = faux - (f1 * 256)
    return np.array([f0, f1, f2], dtype=np.uint8)

def colorform(f):
    return f.reshape(f.shape[0], np.prod(f.shape[-2:]))

def xyform(f):
    x, y = np.indices(f.shape[-2:])
    return np.vstack((x.ravel(), y.ravel())).astype(np.uint16)

def fxyform(f):
    return np.vstack((colorform(f), xyform(f)))