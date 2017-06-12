import unittest
from colorsegmentation.image.ColorSpaceService import *
import numpy as np

class ColorSpaceServiceTest(unittest.TestCase):
    def setUp(self):
        self.f = np.array([[[0],[255],[255],[0],[0],[255],[0],[255],[192],[128],[128],[128],[0],[128],[0],[0]],
                           [[0],[255],[0],[255],[0],[255],[255],[0],[192],[128],[0],[128],[128],[0],[128],[0]],
                           [[0],[255],[0],[0],[255],[0],[255],[255],[192],[128],[0],[0],[0],[128],[128],[128]]], dtype=np.uint8)
    
    def testConvertRGBtoHSV(self):
        g = np.array([[[0],[0],[0],[120],[240],[60],[180],[300],[0],[0],[0],[60],[120],[300],[180],[240]],
                      [[0],[0],[65535],[65535],[65535],[65535],[65535],[65535],[0],[0],[65535],[65535],[65535],[65535],[65535],[65535]],
                      [[0],[65535],[65535],[65535],[65535],[65535],[65535],[65535],[49344],[32896],[32896],[32896],[32896],[32896],[32896],[32896]]], dtype=np.uint16)
        
        self.assertEquals(np.sum(convertRGBtoHSV(self.f) == g), 3 * 16)

suite = unittest.TestLoader().loadTestsFromTestCase(ColorSpaceServiceTest)
unittest.TextTestRunner(verbosity=2).run(suite)