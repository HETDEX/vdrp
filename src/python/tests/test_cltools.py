import os
import subprocess
import sys
from vdrp import cltools
import vdrp
import os
import shutil
import unittest
import tempfile
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
            fnout = "radec0.dat"
            radius = 11.
            ra_offset  = 0.
            dec_offset = 0.
            ra_offset0, dec_offset0 = 0., 0.
            ra_offset1, dec_offset1 = cltools.getoff2(fnradec, fnshuffle_ifustars, radius, fnout, ra_offset0, dec_offset0, logging=None)
            ra_offset2, dec_offset2 = cltools.getoff2(fnradec, fnshuffle_ifustars, radius, fnout, ra_offset1, dec_offset1, logging=None)
            ra_offset3, dec_offset3 = cltools.getoff2(fnradec, fnshuffle_ifustars, radius, fnout, ra_offset2, dec_offset2, logging=None)
            # Check for convergence.
            self.assertAlmostEqual(ra_offset2, ra_offset3)
            self.assertAlmostEqual(dec_offset2, dec_offset3)
            self.assertTrue(fnout)
            print("Files after: ", glob.glob("*"))

