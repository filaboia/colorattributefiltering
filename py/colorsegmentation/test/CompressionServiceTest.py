import unittest

import numpy as np
from colorsegmentation.image.CompressionService import condensa, descondensa


class CompressionServiceTest(unittest.TestCase):
    def setUp(self):
        self.a1 = np.array([0, 1, 2, 3], np.uint8)
        self.a2 = np.array([0, 1, 2, 3, 4], np.uint8)
        self.a3 = np.array([3, 2, 1, 0], np.uint8)
        self.a4 = np.array([3, 2, 1], np.uint8)

        self.c1 = np.array([27], np.uint8)
        self.c2 = np.array([5, 56], np.uint8)
        self.c3 = np.array([228], np.uint8)
        self.c4 = np.array([228], np.uint8)

    def testcondensedivisible(self):
        c = condensa(self.a1, 2)
        self.assertEquals(np.sum(c == self.c1), 1)
    
    def testcondensenotdivisible(self):
        c = condensa(self.a2, 3)
        self.assertEquals(np.sum(c == self.c2), 2)
    
    def testcondenserightzeros(self):
        c = condensa(self.a3, 2)
        self.assertEquals(np.sum(c == self.c3), 1)
    
    def testcondensenotrightzeros(self):
        c = condensa(self.a4, 2)
        self.assertEquals(np.sum(c == self.c4), 1)
    
    def testuncondensedivisible(self):
        a = descondensa(self.c1, 2, 4)
        self.assertEquals(np.sum(a == self.a1), 4)
    
    def testuncondensenotdivisible(self):
        a = descondensa(self.c2, 3, 5)
        self.assertEquals(np.sum(a == self.a2), 5)
    
    def testuncondenserightzeros(self):
        a = descondensa(self.c3, 2, 4)
        self.assertEquals(np.sum(a == self.a3), 4)
    
    def testuncondensenotrightzeros(self):
        a = descondensa(self.c4, 2, 3)
        self.assertEquals(np.sum(a == self.a4), 3)

suite = unittest.TestLoader().loadTestsFromTestCase(CompressionServiceTest)
unittest.TextTestRunner(verbosity=2).run(suite)
