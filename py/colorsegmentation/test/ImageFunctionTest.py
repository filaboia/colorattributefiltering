import unittest
from colorsegmentation.image.ImageFunction import *
from colorsegmentation.image.ImageService import fxyform, codifica, colorform
from adpil import adread

class ImageFunctionTest(unittest.TestCase):
    def setUp(self):
        self.f = adread("../images/test/circulo2.png")
        self.w = adread("../images/test/circulo2seg.png")
    
    def testArea(self):
        self.assertEquals(Area(colorform(self.f)[0]), 25)
    
    def testEntropy(self):
        self.assertAlmostEqual(Entropy(codifica(self.f).ravel()), 1.6430741894285699)
    
    def testAverageColorError(self):
        self.assertAlmostEqual(AverageColorError(colorform(self.f)), 2462.29, delta=0.01)
    
    def testColorHarmony(self):
        self.assertAlmostEqual(ColorHarmony(colorform(self.f)), -0.18, delta=0.01)

suite = unittest.TestLoader().loadTestsFromTestCase(ImageFunctionTest)
unittest.TextTestRunner(verbosity=2).run(suite)