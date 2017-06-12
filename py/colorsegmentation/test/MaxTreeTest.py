import unittest
from colorsegmentation.maxtree.MaxTree import MaxTree
from colorsegmentation.image.ImageFunction import Area, Entropy
from colorsegmentation.image.ImageService import codifica, ordena_minimos
import numpy as np
from adpil import adread, adwrite

class MaxTreeTest(unittest.TestCase):
    def setUp(self):
        self.f = adread("../images/test/circulo2.png")
        self.g = adread("../images/test/circulo2grad.png")
        self.ft = MaxTree(self.g)
    
    def testTreeCreation(self):
        maxaltura, naofolha, folha = self.ft.status()
        
        self.assertEquals(maxaltura, 4)
        self.assertEquals(naofolha, 7)
        self.assertEquals(folha, 4)
    
    def testTreeStructAtt(self):
        ft = self.ft.computeAtt(Area)
        mins = ft.getRegMax(False)
        
        gabMins = adread("../images/test/circulo2gradareaMamba.png")
        
        self.assertEquals(np.sum(mins == gabMins), 12)
    
    def testTreeAtt(self):
        ft = self.ft.computeAtt(Entropy, codifica(self.f))
        mins = (ft.getRegMax(False) * 255 / 24).astype(np.uint8)
        ordMins = ordena_minimos(mins)
        
        gabMins = adread("../images/test/circulo2gradentropiaMamba.png")
        gabOrdMins = adread("../images/test/circulo2gradentropiaordMamba.png")
        
        self.assertEquals(np.sum(mins == gabMins), 12)
        self.assertEquals(np.sum(ordMins == gabOrdMins), 25)
    
    def testTreeRegAtt(self):
        ft = self.ft.computeRegAtt(Entropy, codifica(self.f))
        mins = (ft.getRegMax(False) * 255 / 24).astype(np.uint8)
        
        gabMins = adread("../images/test/circulo2gradentropiaregMamba.png")
        
        self.assertEquals(np.sum(mins == gabMins), 12)
        
suite = unittest.TestLoader().loadTestsFromTestCase(MaxTreeTest)
unittest.TextTestRunner(verbosity=2).run(suite)