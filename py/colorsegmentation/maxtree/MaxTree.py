# This Python file uses the following encoding: utf-8
from __future__ import division
from colorsegmentation.image.ImageService import *
from colorsegmentation.image.Points import *
import numpy as np
import numpy.ma as ma
import copy as cp
from morph import *
import sys

sys.setrecursionlimit(10000) 

class MaxTree(object):
    def __init__(self, f=None):
        self.children = []
        self.level = None
        self.index = None
        self.att = None

        if f is None:
            self.mask = Points(np.uint16)
            return

        self.mask = Points(np.uint16, f >= 0)

        toprocess = [self]
        index = np.zeros(f.max() + 1, dtype=int)

        while toprocess:
            node = toprocess.pop()
            node.level = f[node.mask()].min()
            node.index = index[node.level]
            index[node.level] += 1

            offset = (node.mask.getX().min(), node.mask.getY().min())
            indices = (node.mask.getX() - offset[0], node.mask.getY() - offset[1])
            region = np.zeros((indices[0].max() + 1, indices[1].max() + 1), dtype = np.bool)
            region[indices] = f[node.mask()] > node.level

            oldregion = np.zeros((indices[0].max() + 1, indices[1].max() + 1), dtype = np.bool)
            oldregion[indices] = f[node.mask()] >= node.level

            node.mask = Points(np.uint16, oldregion - region)
            node.mask.setX(node.mask.getX() + offset[0])
            node.mask.setY(node.mask.getY() + offset[1])

            label = mmlabel(region)

            if len(label.shape) == 1:
                label = label[np.newaxis]

            for i in range(1, label.max() + 1):
                child = MaxTree()
                node.children.append(child)
                child.mask = Points(np.uint16, label == i)
                child.mask.setX(child.mask.getX() + offset[0])
                child.mask.setY(child.mask.getY() + offset[1])
                toprocess.append(child)

    def getKey(self):
        return "%d.%d" % (self.level, self.index)

    def simplify(self):
        ft = cp.deepcopy(self)

        toprocess = [ft]

        while toprocess:
            node = toprocess.pop()
            schild = node
            while len(schild.children) == 1:
                schild = schild.children[0]
                node.mask.merge(schild.mask)
            node.children = schild.children
            toprocess += node.children

        return ft

    def getOffset(self):
        xmin = float("inf")
        xmax = float("-inf")
        ymin = float("inf")
        ymax = float("-inf")

        toprocess = [self]

        while toprocess:
            node = toprocess.pop()
            toprocess += node.children
            if node.mask.getX().min() < xmin:
                xmin = node.mask.getX().min()
            if node.mask.getX().max() > xmax:
                xmax = node.mask.getX().max()
            if node.mask.getY().min() < ymin:
                ymin = node.mask.getY().min()
            if node.mask.getY().max() > ymax:
                ymax = node.mask.getY().max()

        return (xmin, xmax, ymin, ymax)

    def getImage(self):
        toprocess = [self]

        xmin, xmax, ymin, ymax = self.getOffset()

        image = np.zeros((xmax - xmin + 1, ymax - ymin + 1), np.uint8)

        while toprocess:
            node = toprocess.pop()
            image[node.mask.getX() - xmin, node.mask.getY() - ymin] = node.level

            toprocess += node.children

        return image

    def getNodes(self, level, index=None):
        nodes = []

        if self.level == level and (index is None or index == self.index):
            nodes.append(self)
        elif self.level < level:
            for child in self.children:
                nodes += child.getNodes(level, index)

        return nodes

    def getRegMax(self, binary=True):
        ft = self.simplify()

        toprocess = [ft]

        xmin, xmax, ymin, ymax = ft.getOffset()
        image = np.zeros((xmax - xmin + 1, ymax - ymin + 1), np.bool if binary else type(ft.att))

        if binary:
            while toprocess:
                node = toprocess.pop()
                toprocess += node.children
                if not node.children:
                    image[node.mask.getX() - xmin, node.mask.getY() - ymin] = 1
        else:
            postorder = []
            while toprocess:
                node = toprocess.pop()
                toprocess += node.children
                postorder.append(node)

            while postorder:
                node = postorder.pop()
                maxchild, maxatt = (None, float("-inf"))
                for child in node.children:
                    if child.att > maxatt:
                        maxchild, maxatt = (child, child.att)
                if maxchild:
                    node.mask = maxchild.mask
                    maxchild.mask = None

            toprocess = [ft]

            while toprocess:
                node = toprocess.pop()
                toprocess += node.children
                if node.mask:
                    image[node.mask.getX() - xmin, node.mask.getY() - ymin] = node.att
            
            image = ma.array(image, mask=~self.getRegMax())
                    
        return image

    def computeAtt(self, ImageFunction, orig=None):
        ft = cp.deepcopy(self)
        
        image = ft.getImage()

        toprocess = [ft]
        postorder = []

        while toprocess:
            node = toprocess.pop()
            toprocess += node.children
            postorder.append(node)

        maskdict = {}

        while postorder:
            node = postorder.pop()
            mask = Points(np.uint16)
            for child in node.children:
                mask.merge(maskdict[child.getKey()])
                del maskdict[child.getKey()]
            mask.merge(node.mask)
            maskdict[node.getKey()] = mask
            if orig is None:
                node.att = ImageFunction(image[mask()])
            else:
                node.att = ImageFunction(orig[..., mask.getX(), mask.getY()])

        return ft

    def computeRegAtt(self, ImageFunction, orig):
        ft = self.simplify()
    
        igrad = ft.getImage()
        grad = mmneg(igrad)
        ws = mmcwatershed(grad, ft.getRegMax(), mmsecross(), 'REGIONS')
    
        preorder = [ft]
        postorder = []
    
        regdict = {}
    
        while preorder:
            node = preorder.pop()
            preorder += node.children
            postorder.append(node)
    
        while postorder:
            node = postorder.pop()
            if node.children:
                mask = Points(np.uint16)
                for child in node.children:
                    mask.merge(regdict[child.getKey()])
                    del regdict[child.getKey()]
            else:
                mask = Points(np.uint16, ws[node.mask()][0] == ws)
            regdict[node.getKey()] = mask
            node.att = ImageFunction(orig[..., mask.getX(), mask.getY()])
            # node.att = ImageFunction(orig[..., mask.getX(), mask.getY()], ws[mask()])
            
        return ft
        
    def computeRegAttFraction(self, ImageFunction, orig):
        ft = self.simplify()
    
        igrad = ft.getImage()
        grad = mmneg(igrad)
        ws = mmcwatershed(grad, ft.getRegMax(), mmsecross(), 'REGIONS')
    
        preorder = [ft]
        postorder = []
    
        regdict = {}
    
        while preorder:
            node = preorder.pop()
            preorder += node.children
            postorder.append(node)
    
        while postorder:
            node = postorder.pop()
            if node.children:
                attList = []
                for child in node.children:
                    attList += regdict[child.getKey()]
                    del regdict[child.getKey()]
            else:
                mask = Points(np.uint16, ws[node.mask()][0] == ws)
                attList = [ImageFunction(orig[..., mask.getX(), mask.getY()])]
            regdict[node.getKey()] = attList
            node.att = np.mean(attList)
            
        return ft    

    def getPoints(self, function):
        toprocess = [self]

        region = Points(np.uint16)

        while toprocess:
            node = toprocess.pop()
            toprocess += node.children

            if function(node):
                region.merge(function(node))

        return region

    def open(self, value):
        ft = cp.deepcopy(self)

        toprocess = [ft]

        while toprocess:
            node = toprocess.pop()
            newchildren = []
            for child in node.children:
                if child.att >= value:
                    newchildren.append(child)
                    toprocess.append(child)
                else:
                    node.mask.merge(child.getPoints(lambda x: x.mask))
            node.children = newchildren

        return ft
    
    def upperOpen(self, value):
        ft = cp.deepcopy(self)

        toprocess = [ft]

        while toprocess:
            node = toprocess.pop()
            if node.att > value:
                toprocess += node.children
            else:
                node.mask = node.getPoints(lambda x: x.mask)
                node.children = []
                
        return ft
        
    def openReg(self, n_reg):
        ft = cp.copy(self)
        ft.children = []
        
        chosenNodes = [ft]
        toprocess = [(ft, node) for node in self.children]
        compareFunction = lambda x,y: cmp(x[1].att, y[1].att)
        toprocess.sort(compareFunction)
        
        while toprocess:
            father, node = toprocess.pop()
            if len(chosenNodes) < n_reg:
                try:
                    chosenNodes.remove(father)
                except ValueError:
                    pass
                
                newNode = cp.copy(node)
                newNode.children = []    
                chosenNodes.append(newNode)
                father.children.append(newNode)
                
                toprocess += [(newNode, child) for child in node.children]
                toprocess.sort(compareFunction)
            else:
                father.mask.merge(node.getPoints(lambda x: x.mask))
        
        return ft
        
    def getRegions(self, n_reg):
        ft = self.simplify()
    
        igrad = ft.getImage()
        grad = mmneg(igrad)
        ws = mmcwatershed(grad, ft.getRegMax(), mmsecross(), 'REGIONS')
        
        toprocess = [ft]
        
        c_reg = 1
        w_reg = 1
        
        xmin, xmax, ymin, ymax = ft.getOffset()
        image = np.zeros((xmax - xmin + 1, ymax - ymin + 1), np.uint16)
        
        while toprocess:
            node = toprocess.pop()
            if node.children:
                if c_reg < n_reg:
                    c_reg += len(node.children) - 1
                    toprocess += node.children
                    toprocess.sort(lambda x,y: -cmp(x.att, y.att))
                else:
                    childrentoprocess = node.children
                    region = Points(np.uint16)
                    while childrentoprocess:
                        child = childrentoprocess.pop()
                        childrentoprocess += child.children
                        if not child.children:
                            region.merge(Points(np.uint16, ws[child.mask()][0] == ws))
                    image[region()] = w_reg
                    w_reg += 1
            else:
                image[Points(np.uint16, ws[node.mask()][0] == ws)] = w_reg
                w_reg += 1
                        
        return image
    
    def equalizeChildAtt(self):
        ft = cp.deepcopy(self)
        
        toprocess = [ft]
        
        while toprocess:
            node = toprocess.pop()
            toprocess += node.children
            maxAtt = float("-inf")
            for child in node.children:
                if child.att > maxAtt:
                    maxAtt = child.att
            for child in node.children:
                child.att = maxAtt
        
        return ft       
    
    def status(self):
        naofolha = 0
        folha = 0
        maxaltura = 0
        toprocess = [(self, 0)]
        while toprocess:
            node, altura = toprocess.pop()
            toprocess += [(child, altura + 1) for child in node.children]
            if altura > maxaltura:
                maxaltura = altura
            if node.children:
                naofolha += 1
            else:
                folha += 1
        
        return (maxaltura, naofolha, folha)