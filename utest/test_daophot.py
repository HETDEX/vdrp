import os
import subprocess
import sys
from vdrp import daophot
import vdrp
import os
import shutil
import unittest
import tempfile
import path
import glob

from testbase import TestBase

class TestDaophotFind(TestBase):
    ff = ["20180611T054545_034.fits", "daophot.opt"]

    def test_daophot_find(self):
        prefix = "20180611T054545_034"
        sigma = 2
        with path.Path(self.test_dir):
            print("Files before: ", glob.glob("*"))
            daophot.daophot_find(prefix, sigma)
            self.assertTrue(os.path.exists(prefix + ".coo"))
            print("Files after: ", glob.glob("*"))

class TestDaophotPhot(TestBase):
    ff = ["20180611T054545_034.fits", "daophot.opt", "photo.opt", "20180611T054545_034.coo"]

    def test_daophot_phot(self):
        prefix = "20180611T054545_034"
        with path.Path(self.test_dir):
            print("Files before: ", glob.glob("*"))
            daophot.daophot_phot(prefix)
            self.assertTrue(os.path.exists(prefix + ".ap"))
            print("Files after: ", glob.glob("*"))


class TestAllstar(TestBase):
    ff = ["20180611T054545_034.fits", "allstar.opt", "use.psf", "20180611T054545_034.ap"]

    def test_allstar(self):
        prefix = "20180611T054545_034"
        psf = "use.psf"
        with path.Path(self.test_dir):
            print("Files before: ",glob.glob("*"))
            daophot.allstar(prefix, psf = psf)
            self.assertTrue(os.path.exists(prefix + ".als"))
            print("Files after: ", glob.glob("*"))


class TestDaomaster(TestBase):
    ff = ["all.mch", "20180611T054545tot.als","20180611T055249tot.als","20180611T060006tot.als"]

    def test_daomaster(self):
        with path.Path(self.test_dir):
            print("Files before: ",glob.glob("*"))
            daophot.daomaster()
            print("Files after: ", glob.glob("*"))
            self.assertTrue(os.path.exists("all.raw"))

class TestFilterDaophot(TestBase):
    ff = ["20180611T055249_053.coo"]

    def test_filter_daophot_out(self):
        with path.Path(self.test_dir):
            print("Files before: ",glob.glob("*"))
            prefix = "20180611T055249_053"
            file_in  = prefix + ".coo"
            file_out = prefix + ".lst"
            xmin,xmax,ymin,ymax = 5,45,5,45
            daophot.filter_daophot_out(file_in, file_out, xmin,xmax,ymin,ymax)
            print("Files after: ", glob.glob("*"))
            # Make sure output files exists.
            self.assertTrue(os.path.exists(file_out))
            # Check that only x y ramin that fall inside of xmin,xmax and ymin,ymax.
            with open(file_out) as f:
                ll = f.readlines()
                for l in ll[3:]:
                    tt = l.split()
                    x,y = float(tt[1]), float(tt[2])
                    self.assertTrue(x > xmin)
                    self.assertTrue(y < xmax)
                    self.assertTrue(y > ymin)
                    self.assertTrue(y < ymax)

class TestDaophotAls(TestBase):
    ff = ["20180611T054545_015.als"]

    def test_daophot_als_read(self):
        with path.Path(self.test_dir):
            als = daophot.DAOPHOT_ALS.read("20180611T054545_015.als")
        self.assertAlmostEqual(als.data["X"][0], 8.146)
        self.assertAlmostEqual(als.data["Y"][0], 15.462)
        self.assertAlmostEqual(als.FRAD, 3.00)
        print(als.data)

if __name__ == '__main__':
    unittest.main()

