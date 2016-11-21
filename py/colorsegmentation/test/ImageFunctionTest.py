import unittest
from colorsegmentation.image.ImageFunction import *
from colorsegmentation.image.ImageService import xyform, fxyform, codifica, colorform
from adpil import adread

class ImageFunctionTest(unittest.TestCase):
    def setUp(self):
        self.f = adread("../images/test/circulo2.png")
        self.w = adread("../images/test/circulo2seg.png")
    
    def testArea(self):
        self.assertEquals(Area(xyform(self.f)), 25)
    
    def testEntropia(self):
        self.assertAlmostEqual(Entropia(codifica(self.f).ravel()), 1.6430741894285699)
    
    def testErroMedioQuadratico(self):
        self.assertAlmostEqual(ErroMedioQuadratico(colorform(self.f)), 259377.12)
    
    def testHarmoniaCor(self):
        self.assertAlmostEqual(HarmoniaCor(fxyform(self.f)), -3.63, delta=0.01)
    
    def testErroMedioQuadraticoPonderado(self):
        self.assertAlmostEqual(ErroMedioQuadraticoPonderado(self.f, self.w), 169153.44)
    
    def testLiuF(self):
        self.assertAlmostEqual(LiuF(self.f, self.w), 87544.494744101408)
    
    def testBorsottiF(self):
        self.assertAlmostEqual(BorsottiF(self.f, self.w), 0.49522644711283348)
    
    def testBorsottiQ(self):
        self.assertAlmostEqual(BorsottiQ(self.f, self.w), 0.62126878475045866)
    
    def testEntropiaPonderada(self):
        self.assertAlmostEqual(EntropiaPonderada(self.f, self.w), 0.92114609454120744)
        
    def testEntropiaVariancia(self):
        self.assertAlmostEqual(EntropiaVariancia(self.f, self.w), 0.015503051322687263)
        
    def testEntropiaSoma(self):
        self.assertAlmostEqual(EntropiaSoma(self.f, self.w), 23.028652363530181)
    
    def testDesordemZhang(self):
        self.assertAlmostEqual(DesordemZhang(self.f, self.w), 1.3026972998271849)
    
    def testEntropiaZhang(self):
        self.assertAlmostEqual(EntropiaZhang(self.f, self.w), 1.6430741894285696)
    
    def testHarmoniaCorPonderada(self):
        self.assertAlmostEqual(HarmoniaCorPonderada(self.f, self.w), -2.40, delta=0.01)
    
    def testHarmoniaCorSegmentada(self):
        self.assertAlmostEqual(HarmoniaCorSegmentada(self.f, self.w), -0.25, delta=0.01)

suite = unittest.TestLoader().loadTestsFromTestCase(ImageFunctionTest)
unittest.TextTestRunner(verbosity=2).run(suite)