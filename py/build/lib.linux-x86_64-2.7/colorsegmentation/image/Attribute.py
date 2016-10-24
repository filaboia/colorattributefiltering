# This Python file uses the following encoding: utf-8
from __future__ import division
from colorsegmentation.image.ImageService import *
import numpy as np

def erro_medio_quadratico(fxy):
    f = fxy[:3].astype(float)
    media = np.mean(f, axis=1,  keepdims=True)
    return np.sum(pow(f - media, 2))
    
def erro_medio_quadratico_reg(f, w):
    # f = f.astype(float)
    # media = filagrain(w, f, 'mean')
    # dist = np.sum(pow(f - media, 2), axis = 0)
    # return filagrain(w, dist, 'sum', 'data')
    return filagrain(w, f, erro_medio_quadratico, 'data', True)

def erro_medio_quadratico_w(f, w):
    w = normaliza(w)
    A = region_area(w)
    NxM = np.sum(A)
    e2 = erro_medio_quadratico_reg(f, w)
    return np.sum(A * e2) / NxM

def normaliza(w):
    return (np.cumsum(np.concatenate(([0], np.bincount(w.ravel())[1:])) > 0)[w]).astype(w.dtype)

def count_regiao(A):
    return np.bincount(A)[A]

def region_area(w):
    return np.bincount(w.ravel())[1:] 

def liu_F(f, w):
    w = normaliza(w)
    A = region_area(w)
    R = w.max()
    e2 = np.squeeze(erro_medio_quadratico_reg(f, w))
    return pow(R, 0.5) * np.sum(e2 / pow(A, 0.5))

def borsotti_F(f, w):
    w = normaliza(w)
    A = region_area(w)
    e2 = np.squeeze(erro_medio_quadratico_reg(f, w))
    R = w.max()
    NxM = np.sum(A)
    return pow(R, 0.5) / (10000 * NxM) * pow(np.sum(pow(np.bincount(A)[1:], 1 + 1 / (np.arange(A.max()) + 1))) , .5) * np.sum(e2 / pow(A, 0.5))

def borsotti_Q(f, w):
    w = normaliza(w)
    A = region_area(w)
    e2 = np.squeeze(erro_medio_quadratico_reg(f, w))
    R = w.max()
    NxM = np.sum(A)
    return pow(R, 0.5) / (10000 * NxM) * np.sum(e2 / (1 + np.log10(A)) + pow(count_regiao(A) / A, 2))
    
def codifica(f):
    f = f.astype(np.uint32)
    return normaliza(f[0] * 65536 + f[1] * 256 + f[2])
    
def entropia(fxy):
    f = fxy[0]
    count = np.bincount(normaliza(f))
    p = count/np.sum(count)
    log = np.log2(p); log[np.isinf(log)] = 0
    return -np.sum(p*log)
    
def entropia_reg(f, w):
    return filagrain(w, codifica(f), entropia, 'data')
    
def entropia_w(f, w):
    w = normaliza(w)
    A = region_area(w)
    NxM = np.sum(A)
    return np.sum(A * np.squeeze(entropia_reg(f, w))) / NxM
    
def desordem_zhang(f, w):
    w = normaliza(w)
    R = w.max()
    return pow(R, 0.5) * entropia_w(f, w)
    
def entropia_zhang(f, w):
    return entropia_w(f, w) + entropia(w.ravel())

def entropia_banda(f):
    return entropia(f[0]) + entropia(f[1]) + entropia(f[2])
        
def entropia_banda_reg(f, w):
    return np.sum(filagrain(w, f, entropia, 'data'), axis=1)
    
def entropia_banda_w(f, w):
    w = normaliza(w)
    A = region_area(w)
    NxM = np.sum(A)
    return np.sum(A * np.squeeze(entropia_banda_reg(f, w))) / NxM

def entropia_zhang_banda(f, w):
    return entropia_banda_w(f, w) + entropia(w.ravel())
    
def harmonia_duas_cores(p1, p2):
    def cab(p):
        return pow(pow(p[1], 2) + pow(p[2], 2), 0.5)
    
    def hab(p):
        return np.arctan2(p[2], p[1]) % (2 * np.pi)
    
    def Hlinha(hab):
        return -0.23 - 0.35 * np.sin(hab + 0.83) - 0.18 * np.sin(2 * hab + 1.55)
    
    def atualiza_range(p):
        novop = np.empty(p.shape, dtype=np.float)    
        novop[0] = p[0] / 255. * 100. 
        novop[1] = p[1] / 255. * 254. - 127.
        novop[2] = p[2] / 255. * 254. - 127.
        return novop.astype(int)
    
    p1 = atualiza_range(p1)
    p2 = atualiza_range(p2)
    Lsum = p1[0] + p2[0]
    deltaCab = abs(cab(p1) - cab(p2))
    deltaL = abs(p1[0] - p2[0])
    deltaa = abs(p1[1] - p2[1])
    deltab = abs(p1[2] - p2[2])
    deltaHab = np.nan_to_num(pow(pow(deltaa, 2) + pow(deltab, 2) - pow(deltaCab, 2), 0.5))
    # deltaEab = pow(pow(deltaL, 2) + pow(deltaa, 2) + pow(deltab, 2), 0.5)
    # deltaHab = np.nan_to_num(pow(pow(deltaEab, 2) - pow(deltaL, 2) - pow(deltaCab, 2), 0.5))
    deltaC = pow(deltaHab + pow(deltaCab / 1.5, 2), 0.5)
    HdeltaC = 1.3 - 0.07 * deltaC + 0.0005 * pow(deltaC, 2)
    HdeltaL = -0.92 + 0.05 * deltaL - 0.0006 * pow(deltaL, 2)
    Hh = Hlinha(hab(p1)) + Hlinha(hab(p2))
    return -2.2 + 0.03 * Lsum + HdeltaC + HdeltaL + 1.1 * Hh

def harmonia_cor(fxy):
    f = fxy[:3].astype(np.uint8)
    x = fxy[3].astype(np.uint16)
    y = fxy[4].astype(np.uint16)
    MxN = f.shape[1]
    
    return np.sum([np.sum(harmonia_duas_cores(np.full_like(f[:, i+1:], f[:, i][:, np.newaxis]), f[:, i+1:]) 
    / pow(pow(np.full_like(x[i+1:], x[i]) - x[i+1:], 2) + pow(np.full_like(y[i+1:], y[i]) - y[i+1:], 2), 0.5)) 
    for i in range(MxN - 1)]) / (2 * MxN)
    
    # def function1(f):
    #     return np.concatenate([np.full_like(f[..., i+1:], f[..., i][..., np.newaxis]) for i in range(MxN - 1)], axis=-1) 
    # 
    # def function2(f):
    #     return np.concatenate([f[..., i+1:] for i in range(MxN - 1)], axis=-1)
    # 
    # return np.sum(harmonia_duas_cores(function1(f), function2(f)) / pow(pow(function1(x) - function2(x), 2) + pow(function1(y) - function2(y), 2), 0.5)) / (2 * MxN)
    
    # def function1(f):
    #     return np.repeat(f, MxN, axis=-1)
    # 
    # def function2(f):
    #     return np.tile(f, MxN)
    # 
    # mask = np.arange(MxN)
    # mask = np.concatenate((function1(mask)[..., np.newaxis], function2(mask)[..., np.newaxis]), axis=-1)
    # mask = mask[...,0] < mask[...,1]
    # 
    # return np.sum(harmonia_duas_cores(function1(f)[..., mask], function2(f)[..., mask]) 
    # / pow(pow(function1(x)[..., mask] - function2(x)[..., mask], 2) 
    # + pow(function1(y)[..., mask] - function2(y)[..., mask], 2), 0.5)) / (2 * MxN)

def harmonia_cor_reg(f, w):
    M, N = w.shape
    x , y = np.indices((M, N))
    f = f.reshape(f.shape[0], M * N)
    x = x.ravel()
    y = y.ravel()
    w = w.ravel().astype(np.uint16)
    fxy = np.vstack((f, np.vstack((x, y)))).astype(np.uint16)
    return filagrain(w[...,np.newaxis], fxy[...,np.newaxis], harmonia_cor, 'data', True)

def harmonia_cor_w(f, w):
    w = normaliza(w)
    A = region_area(w)
    NxM = np.sum(A)
    return np.sum(A * np.squeeze(harmonia_cor_reg(f, w))) / NxM
    
def harmonia_cor_seg(f, w):
    s = filagrain(w, f, 'mean').astype(f.dtype)
    return harmonia_cor_w(s, (w >= 0).astype(w.dtype))
