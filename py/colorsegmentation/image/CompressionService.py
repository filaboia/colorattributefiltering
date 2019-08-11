# This Python file uses the following encoding: utf-8
from __future__ import division
import numpy as np

def complementomodulo(dividendo, divisor):
    return (divisor - dividendo % divisor) % divisor

def condensa(f, bits):
    total = f.size * bits
    compl = complementomodulo(total, 8)
    n = (total + compl) // 8
    return np.packbits(np.pad(np.unpackbits(f[:,np.newaxis], axis=1)[:,-bits:].ravel(), (0, compl), 'constant').reshape(n, 8), axis=1).ravel()

def descondensa(f, bits, size):
    total = size * bits
    return np.packbits(np.pad(np.unpackbits(f[:,np.newaxis], axis=1).ravel()[:total].reshape(size, bits), ((0, 0), (8-bits, 0)), 'constant'), axis=1).ravel()