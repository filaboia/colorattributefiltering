from __future__ import division
import numpy as np

def weightedHueHistogram(f):
    f = f.astype(np.float)
    H = f[0].astype(np.float)
    SV = f[1].astype(np.float) * f[2].astype(np.float)
    
    if np.sum(SV) == 0:
        return np.zeros(360, np.uint8)
    
    return np.sum(np.sum((np.tile(H, (360, 1, 1)) == np.arange(360)[:, np.newaxis, np.newaxis]) * SV[np.newaxis], axis=2), axis=1) / np.sum(SV)

def kullbackLieblerDivergence(h1, h2):
    r = h1 / h2
    r[r != 0] = np.log(r[r != 0])
    return np.sum(h1 * r, axis=-1)

def isInAngularInterval(i, a, w):
    lb = a - w/2
    lb[lb < 0] = lb[lb < 0] + 360
    hb = a + w/2
    hb[hb > 359] = hb[hb > 359] - 360
    
    p = lb < hb
    pn = p == 0
    q = i > lb
    r = i < hb
    
    return q * r + pn * (q + r) > 0

def angularDifference(a, b):
    difference = abs(a - b)
    return abs((difference > 180) * 360 - difference)

def rotate(template, a):
    template[0] = (template[0] + a) % 360
    return template

def iType(a):
    return rotate(np.array([[90],[22.5]], dtype=np.float), a)

def VType(a):
    return rotate(np.array([[90],[90]], dtype=np.float), a)

def LType(a):
    return rotate(np.array([[0,90],[90,22.5]], dtype=np.float), a)

def JType(a):
    return rotate(np.array([[90,180],[22.5,90]], dtype=np.float), a)

def IType(a):
    return rotate(np.array([[90,270],[22.5,22.5]], dtype=np.float), a)

def TType(a):
    return rotate(np.array([[0],[180]], dtype=np.float), a)

def YType(a):
    return rotate(np.array([[90,270],[90,22.5]], dtype=np.float), a)

def XType(a):
    return rotate(np.array([[90,270],[90,90]], dtype=np.float), a)

def templatesDistribution(templateFunctions):
    def sectorsDistribution(template):
        distribution = np.empty((template.shape[1], 360), dtype=np.float)
        i = np.arange(360, dtype=np.float)
        
        mask = isInAngularInterval(i[np.newaxis], template[0][:,np.newaxis], template[1][:,np.newaxis])
        
        distribution[mask == 0] = 0
        distribution[mask] = np.exp(-1/(1-np.power(2*abs(angularDifference(i, template[0][:,np.newaxis]))[mask]/np.tile(template[1][:,np.newaxis], (1, 360))[mask], 10)))
        
        return np.sum(distribution, axis=0)
    
    distribution = np.array([sectorsDistribution(f(a)) for f in templateFunctions for a in range(360)])
    mask = distribution == 0
    distribution[mask] = 0.01 / 360 * np.tile(np.sum(distribution, axis=1, keepdims=True), (1, 360))[mask]
    
    return distribution / np.sum(distribution, axis=1, keepdims=True)

def templateRecognition(h, distribution):
    def decodeTemplate(index):
        return index // 360, index % 360
    
    return decodeTemplate(np.argmin(kullbackLieblerDivergence(h[np.newaxis], distribution)))