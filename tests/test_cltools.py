import os
import sys
from vdrp import cltools
import vdrp
import shutil
import unittest
import path
import glob

from testbase import TestBase

class TestGetoff2(TestBase):
    ff = ["tmp.csv", "shout.ifustars"]

    def test_getoff2(self):
        with path.Path(self.test_dir):
            print("Files before: ", glob.glob("*"))
            fnradec = "tmp.csv"
            fnshuffle_ifustars = "shout.ifustars"
            radius = 11.
            ra_offset  = 0.
            dec_offset = 0.
            ra_offset0, dec_offset0 = 0., 0.
            ra_offset1, dec_offset1 = cltools.getoff2(fnradec, fnshuffle_ifustars, radius, ra_offset0, dec_offset0, logging=None)
            ra_offset2, dec_offset2 = cltools.getoff2(fnradec, fnshuffle_ifustars, radius, ra_offset1, dec_offset1, logging=None)
            ra_offset3, dec_offset3 = cltools.getoff2(fnradec, fnshuffle_ifustars, radius, ra_offset2, dec_offset2, logging=None)
            print("Files after: ", glob.glob("*"))
            # Check for convergence.
            self.assertAlmostEqual(ra_offset2, ra_offset3)
            self.assertAlmostEqual(dec_offset2, dec_offset3)


class TestImmosaicv(TestBase):
    ff = ['20180611T054545_015.fits', '20180611T054545_022.fits',
          '20180611T054545_023.fits', '20180611T054545_024.fits',
          '20180611T054545_026.fits', '20180611T054545_027.fits',
          '20180611T054545_032.fits', '20180611T054545_033.fits',
          '20180611T054545_034.fits', '20180611T054545_035.fits',
          '20180611T054545_036.fits', '20180611T054545_037.fits',
          '20180611T054545_042.fits', '20180611T054545_043.fits',
          '20180611T054545_044.fits', '20180611T054545_045.fits',
          '20180611T054545_046.fits', '20180611T054545_047.fits',
          '20180611T054545_052.fits', '20180611T054545_053.fits',
          '20180611T054545_057.fits', '20180611T054545_062.fits',
          '20180611T054545_063.fits', '20180611T054545_072.fits',
          '20180611T054545_073.fits', '20180611T054545_074.fits',
          '20180611T054545_075.fits', '20180611T054545_076.fits',
          '20180611T054545_082.fits', '20180611T054545_083.fits',
          '20180611T054545_084.fits', '20180611T054545_085.fits',
          '20180611T054545_086.fits', '20180611T054545_093.fits',
          '20180611T054545_094.fits', '20180611T054545_095.fits',
          '20180611T054545_096.fits', '20180611T054545_103.fits',
          '20180611T054545_104.fits', '20180611T054545_106.fits',\
          'fplane.txt']

    def test_immosaicv(self):
        with path.Path(self.test_dir):

            print("Files before: ", glob.glob("*"))
            prefixes = ['20180611T054545_015', '20180611T054545_022',
                        '20180611T054545_023', '20180611T054545_024',
                        '20180611T054545_026', '20180611T054545_027',
                        '20180611T054545_032', '20180611T054545_033',
                        '20180611T054545_034', '20180611T054545_035',
                        '20180611T054545_036', '20180611T054545_037',
                        '20180611T054545_042', '20180611T054545_043',
                        '20180611T054545_044', '20180611T054545_045',
                        '20180611T054545_046', '20180611T054545_047',
                        '20180611T054545_052', '20180611T054545_053',
                        '20180611T054545_057', '20180611T054545_062',
                        '20180611T054545_063', '20180611T054545_072',
                        '20180611T054545_073', '20180611T054545_074',
                        '20180611T054545_075', '20180611T054545_076',
                        '20180611T054545_082', '20180611T054545_083',
                        '20180611T054545_084', '20180611T054545_085',
                        '20180611T054545_086', '20180611T054545_093',
                        '20180611T054545_094', '20180611T054545_095',
                        '20180611T054545_096', '20180611T054545_103',
                        '20180611T054545_104', '20180611T054545_106']
            cltools.immosaicv(prefixes, "fplane.txt")
            print("Files after: ", glob.glob("*"))
            # Make sure output files exists.
            fnout = "immosaic.fits"
            self.assertTrue(os.path.exists(fnout))


class TestImrot(TestBase):
    ff = ['immosaic.fits']

    def test_imrot(self):
        with path.Path(self.test_dir):
            print("Files before: ", glob.glob("*"))
            angle = 42.
            cltools.imrot('immosaic.fits', angle, logging=None)
            print("Files after: ", glob.glob("*"))
            # Make sure output files exists.
            fnout = "imrot.fits"
            self.assertTrue(os.path.exists(fnout))

