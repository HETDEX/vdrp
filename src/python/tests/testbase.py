import unittest
import tempfile
import os
import shutil
import vdrp

class TestBase(unittest.TestCase):
    """
    Provides funtionality to set up test directory with
    predifined list of input data.
    """
    ff = []
    dd = []

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

        for d in cls.dd:
            shutil.copytree(os.path.join(ptestdata,d), os.path.join(cls.test_dir, d))

    @classmethod
    def tearDownClass(cls):
        print("Tearing class down.")
        # Remove the directory after the test
        shutil.rmtree(cls.test_dir)
        #print("WARNING: NOT REMOVING TEMPORARY DIRECTORY")



