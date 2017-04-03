# This Python file uses the following encoding: utf-8
from __future__ import division
import numpy as np
import numpy.ma as ma
import pandas as pd
from ia870 import *

def harmony(p1, p2):
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
    
    return ma.masked_invalid(HC + HL + HH)
            
def euclid(f, fd):
    return pow(np.sum(pow(f - fd, 2), axis=0),0.5)
    
def taxi(f, fd):
    return np.sum(abs(f - fd), axis=0)
    
def chess(f, fd):
    return np.max(abs(f - fd), axis=0)
    
def identity(f, fd=None):
    return f
    
def gradient(f, gradientType=1, distanceType='euclid', includeOrigin=False, normalize=True):
    def desloca(f, dx, dy):
        g = np.copy(f)
        zmax, xmax, ymax = f.shape
        g[:,np.clip(dx, 0, xmax):np.clip(xmax+dx, 0, xmax),np.clip(dy, 0, ymax):np.clip(ymax+dy, 0, ymax)] = f[:,np.clip(-dx, 0, xmax):np.clip(xmax-dx, 0, xmax),np.clip(-dy, 0, ymax):np.clip(ymax-dy, 0, ymax)]
        return g
    
    try:    
        distance = globals()[distanceType]
    except:
        distance = identity
    
    fdtype = f.dtype
    max_dtype = np.iinfo(fdtype).max
    
    f = f.astype(np.float)
    
    distances = [distance(f, desloca(f, 0, 1)), distance(f, desloca(f, 0, -1)), distance(f, desloca(f, 1, 0)), distance(f, desloca(f, -1, 0))]
    
    if includeOrigin:
        distances += [distance(f, f)]
    
    if gradientType == 1:
        g = ma.max(distances, axis=0) 
    elif gradientType == 2:
        g = ma.max(distances, axis=0) - ma.min(distances, axis=0) 
    elif gradientType == 3:
        distances += [distance(desloca(f, 0, 1), desloca(f, 0, -1)), distance(desloca(f, 0, 1), desloca(f, 1, 0)), distance(desloca(f, 0, 1), desloca(f, -1, 0))]
        distances += [distance(desloca(f, 0, -1), desloca(f, 1, 0)), distance(desloca(f, 0, -1), desloca(f, -1, 0))]
        distances += [distance(desloca(f, 1, 0), desloca(f, -1, 0))]
        g = ma.max(distances, axis=0) 
    elif gradientType == 4:
        g = ma.min(distances, axis=0) 
    elif gradientType == 5:
        distances += [distance(desloca(f, 0, 1), desloca(f, 0, -1)), distance(desloca(f, 0, 1), desloca(f, 1, 0)), distance(desloca(f, 0, 1), desloca(f, -1, 0))]
        distances += [distance(desloca(f, 0, -1), desloca(f, 1, 0)), distance(desloca(f, 0, -1), desloca(f, -1, 0))]
        distances += [distance(desloca(f, 1, 0), desloca(f, -1, 0))]
        g = ma.min(distances, axis=0) 
    else:
        raise ValueError
    
    if normalize:
        g = (g - g.min()) / (g.max() - g.min()) * max_dtype   
        g = np.clip(np.round(g), 0, max_dtype).astype(fdtype)    
        
    return g
    
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
    reg_mins = grain(ialabel(~mins.mask), o_mins, 'min')
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