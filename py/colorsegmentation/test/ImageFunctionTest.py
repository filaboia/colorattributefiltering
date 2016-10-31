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
    
    def testEntropiaBanda(self):
        self.assertAlmostEqual(EntropiaBanda(colorform(self.f)), 4.7480495677664774)
    
    def testErroMedioQuadratico(self):
        self.assertAlmostEqual(ErroMedioQuadratico(colorform(self.f)), 259377.12)
    
    def testHarmoniaCor(self):
        self.assertAlmostEqual(HarmoniaCor(fxyform(self.f)), -3.63, delta=0.01)
    
    def testErroMedioQuadraticoWeighted(self):
        self.assertAlmostEqual(ErroMedioQuadraticoWeighted(self.f, self.w), 169153.44)
    
    def testLiuF(self):
        self.assertAlmostEqual(LiuF(self.f, self.w), 87544.494744101408)
    
    def testBorsottiF(self):
        self.assertAlmostEqual(BorsottiF(self.f, self.w), 0.49522644711283348)
    
    def testBorsottiQ(self):
        self.assertAlmostEqual(BorsottiQ(self.f, self.w), 0.62126878475045866)
    
    def testEntropiaWeighted(self):
        self.assertAlmostEqual(EntropiaWeighted(self.f, self.w), 0.92114609454120744)
    
    def testDesordemZhang(self):
        self.assertAlmostEqual(DesordemZhang(self.f, self.w), 1.3026972998271849)
    
    def testEntropiaZhang(self):
        self.assertAlmostEqual(EntropiaZhang(self.f, self.w), 1.6430741894285696)
    
    def testEntropiaBandaWeighted(self):
        self.assertAlmostEqual(EntropiaBandaWeighted(self.f, self.w), 2.7634382836236218)
    
    def testEntropiaZhangBanda(self):
        self.assertAlmostEqual(EntropiaZhangBanda(self.f, self.w), 3.4853663785109843)
    
    def testHarmoniaCorWeighted(self):
        self.assertAlmostEqual(HarmoniaCorWeighted(self.f, self.w), -2.40, delta=0.01)
    
    def testHarmoniaCorSegmented(self):
        self.assertAlmostEqual(HarmoniaCorSegmented(self.f, self.w), -0.25, delta=0.01)

suite = unittest.TestLoader().loadTestsFromTestCase(ImageFunctionTest)
unittest.TextTestRunner(verbosity=2).run(suite)