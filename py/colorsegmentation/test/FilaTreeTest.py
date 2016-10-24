import unittest
from colorsegmentation.filatree.FilaTree import FilaTree
from colorsegmentation.image.ImageFunction import Area, Entropia, codifica
import numpy as np
from adpil import adread, adwrite

class FilaTreeTest(unittest.TestCase):
    def setUp(self):
        self.f = adread("../images/test/circulo2.png")
        self.g = adread("../images/test/circulo2grad.png")
        self.ft = FilaTree(self.g)
    
    def testTreeCreation(self):
        maxaltura, naofolha, folha = self.ft.status()
        
        self.assertEquals(maxaltura, 4)
        self.assertEquals(naofolha, 7)
        self.assertEquals(folha, 8)
    
    def testTreeStructAtt(self):
        ft = self.ft.computeAtt(Area)
        mins = ft.getRegMax(False)
        
        gabMins = adread("../images/test/circulo2gradarea.png")
        
        self.assertEquals(np.sum(mins == gabMins), 25)
    
    def testTreeAtt(self):
        ft = self.ft.computeAtt(Entropia, codifica(self.f))
        mins = (ft.getRegMax(False) * 255 / 24).astype(np.uint8)
        
        gabMins = adread("../images/test/circulo2gradentropia.png")
        
        self.assertEquals(np.sum(mins == gabMins), 25)
    
    def testTreeRegAtt(self):
        ft = self.ft.computeRegAtt(Entropia, codifica(self.f))
        mins = (ft.getRegMax(False) * 255 / 24).astype(np.uint8)
        
        gabMins = adread("../images/test/circulo2gradentropiareg.png")
        
        self.assertEquals(np.sum(mins == gabMins), 25)
        
    
suite = unittest.TestLoader().loadTestsFromTestCase(FilaTreeTest)
unittest.TextTestRunner(verbosity=2).run(suite)