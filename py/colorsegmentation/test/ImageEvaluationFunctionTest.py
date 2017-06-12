import unittest
from colorsegmentation.image.ImageEvaluationFunction import *
from colorsegmentation.image.ImageService import xyform, fxyform, codifica, colorform
from adpil import adread

class ImageEvaluationFunctionTest(unittest.TestCase):
    def setUp(self):
        self.f = adread("../images/test/circulo2.png")
        self.w = adread("../images/test/circulo2seg.png")
    
    def testAverageColorErrorWeighted(self):
        self.assertAlmostEqual(AverageColorErrorWeighted(self.f, self.w), 1645.04, delta=0.01)
    
    def testLiuF(self):
        self.assertAlmostEqual(LiuF(self.f, self.w), 1302692.44, delta=0.01)
    
    def testBorsottiF(self):
        self.assertAlmostEqual(BorsottiF(self.f, self.w), 7.37, delta=0.01)
    
    def testBorsottiQ(self):
        self.assertAlmostEqual(BorsottiQ(self.f, self.w), 9.94, delta=0.01)
    
    def testEntropyWeighted(self):
        self.assertAlmostEqual(EntropyWeighted(self.f, self.w), 0.92114609454120744)
        
    def testEntropyVariance(self):
        self.assertAlmostEqual(EntropyVariance(self.f, self.w), 0.015503051322687263)
        
    def testEntropySum(self):
        self.assertAlmostEqual(EntropySum(self.f, self.w), 23.028652363530181)
    
    def testZhangHw(self):
        self.assertAlmostEqual(ZhangHw(self.f, self.w), 1.3026972998271849)
    
    def testZhangE(self):
        self.assertAlmostEqual(ZhangE(self.f, self.w), 13.06, delta=0.01)
    
    def testColorHarmonyWeighted(self):
        self.assertAlmostEqual(ColorHarmonyWeighted(self.f, self.w), 0.46, delta=0.05)
    
    def testColorHarmonySegmented(self):
        self.assertAlmostEqual(ColorHarmonySegmented(self.f, self.w), -0.07, delta=0.01)

suite = unittest.TestLoader().loadTestsFromTestCase(ImageEvaluationFunctionTest)
unittest.TextTestRunner(verbosity=2).run(suite)