import unittest
from colorsegmentation.image.StatisticsService import *

class StatisticsServiceTest(unittest.TestCase):
    def setUp(self):
        self.f = np.array([np.arange(360, dtype=np.uint16), np.linspace(0, 65535, 360, dtype=np.uint16), np.linspace(0, 65535, 360, dtype=np.uint16)])
        
    def testWeightedHueHistogram(self):
        g = np.arange(360, dtype=np.float)
        g = g / np.sum(g)
        
        self.assertAlmostEquals(np.sum(np.round(weightedHueHistogram(self.f), 1) == np.round(g, 1)), 360,)
    
    def testTemplateRecognition(self):
        templateFunctions = [iType, VType, LType, JType, IType, TType, YType, XType]
        distribution = templatesDistribution(templateFunctions)
        index = templateRecognition(templatesDistribution([XType])[47], distribution)
        self.assertEquals(index[0], 7)
        self.assertEquals(index[1], 47)
    
suite = unittest.TestLoader().loadTestsFromTestCase(StatisticsServiceTest)
unittest.TextTestRunner(verbosity=2).run(suite)