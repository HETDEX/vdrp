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

class TestDaophot(unittest.TestCase):
    """
    Provides funtionality to set up test directory with
    predifined list of input data.
    """
    ff = []

    @classmethod
    def setUpClass(cls):

        print("Setting class up.")
        # Create a temporary directory
        cls.test_dir = tempfile.mkdtemp()
        print("Testdir is {}".format(cls.test_dir))
        print("Copy test data...")
        p = vdrp.__path__
        ptestdata = os.path.join(p[0], '../tests/testdata')
        ptestdata = os.path.realpath(ptestdata)
        for f in cls.ff:
            shutil.copy2(os.path.join(ptestdata,f), cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        print("Tearing class down.")
        # Remove the directory after the test
        shutil.rmtree(cls.test_dir)
        #print("WARNING: NOT REMOVIN TEMPORARY DIRECTORY")


class TestDaophotFind(TestDaophot):
    ff = ["20180611T054545_034.fits", "daophot.opt"]

    def test_daophot_find(self):
        prefix = "20180611T054545_034"
        sigma = 2
        with path.Path(self.test_dir):
            print(glob.glob("*"))
            daophot.daophot_find(prefix, sigma)
            self.assertTrue(os.path.exists(prefix + ".coo"))
            print(glob.glob("*"))

class TestDaophotPhot(TestDaophot):
    ff = ["20180611T054545_034.fits", "daophot.opt", "photo.opt", "20180611T054545_034.coo"]

    def test_daophot_phot(self):
        prefix = "20180611T054545_034"
        with path.Path(self.test_dir):
            print(glob.glob("*"))
            daophot.daophot_phot(prefix)
            self.assertTrue(os.path.exists(prefix + ".ap"))
            print(glob.glob("*"))


class TestAllstar(TestDaophot):
    ff = ["20180611T054545_034.fits", "allstar.opt", "use.psf", "20180611T054545_034.ap"]

    def test_allstar(self):
        prefix = "20180611T054545_034"
        psf = "use.psf"
        with path.Path(self.test_dir):
            print(glob.glob("*"))
            daophot.allstar(prefix, psf)
            self.assertTrue(os.path.exists(prefix + ".als"))
            print(glob.glob("*"))


class TestDeomaster(TestDaophot):
    ff = ["all.mch", "20180611T054545tot.als"]

    def test_allstar(self):
        with path.Path(self.test_dir):
            print(glob.glob("*"))
            daophot.daomaster()
            print(glob.glob("*"))


if __name__ == '__main__':
    unittest.main()

