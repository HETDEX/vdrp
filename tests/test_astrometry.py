import os
import sys
from vdrp import cltools
import vdrp
import shutil
import unittest
import path
import glob
from argparse import Namespace

from testbase import TestBase

from vdrp import astrometry
from vdrp.utils import read_radec
from vdrp.astrometry import main
import glob
import path

class Test_Get_Exposures_files(TestBase):
    ff = ["20180611T054545_015.fits", "20180611T055249_015.fits", "20180611T060006_015.fits"]

    def test_get_exposures_files(self):
        with path.Path(self.test_dir):
            exposures_files = astrometry.get_exposures_files(self.test_dir)
            self.assertTrue(("exp01" in exposures_files))
            self.assertTrue(("exp02" in exposures_files))
            self.assertTrue(("exp03" in exposures_files))
            if "exp01" in exposures_files:
                self.assertTrue(exposures_files["exp01"][0] == "20180611T054545_015")
            if "exp02" in exposures_files:
                self.assertTrue(exposures_files["exp02"][0] == "20180611T055249_015")
            if "exp03" in exposures_files:
                self.assertTrue(exposures_files["exp03"][0] == "20180611T060006_015")

class Test_Cp_Post_Stamps(TestBase):
    dd = ["reductions"]

    def test_cp_post_stamps(self):
        with path.Path(self.test_dir):
            night = "20180611"
            shotid = "017"
            astrometry.cp_post_stamps(self.test_dir, "reductions", night, shotid)
            files = ["20180611T054545_015.fits", "20180611T055249_015.fits", "20180611T060006_015.fits"]
            for f in files:
                self.assertTrue( os.path.exists(f) )

class Test_Main(TestBase):
    dd = ["reductions", "config", "test_fiducial"]
    ff = []
    delete_test_dir = False

    def test_main(self):
        with path.Path(self.test_dir):
            args = Namespace
            args.acam_magadd=5.0
            args.add_radec_angoff=1.45
            args.add_radec_angoff_trial=[1.425, 1.45, 1.475, 1.5, 1.525]
            args.add_radec_angoff_trial_dir='add_radec_angoff_trial'
            args.addin_dir='config'
            args.cofes_vis_vmax=25.0
            args.cofes_vis_vmin=-15.0
            #conf_file=None
            args.config_source='config/vdrp.config'
            args.daophot_allstar_opt='config/allstar.opt'
            args.daophot_opt='config/daophot.opt'
            args.daophot_phot_psf='config/use.psf'
            args.daophot_photo_opt='config/photo.opt'
            args.daophot_sigma=2.0
            args.daophot_xmax=45
            args.daophot_xmin=4
            args.daophot_ymin=4
            args.daophot_ymix=45
            args.dec=51.3479
            args.fluxnorm_mag_max=19.0
            args.fplane_txt='config/fplane.txt'
            args.ixy_dir='config'
            args.addin_dir='config'
            args.getoff2_radii=[11.0, 5.0, 3.0]
            args.logfile='20180611v017.log'
            args.mkmosaic_angoff=1.8
            args.mktot_ifu_grid='config/ifu_grid.txt'
            args.mktot_magmax=21.0
            args.mktot_magmin=0.0
            args.mktot_xmax=50.0
            args.mktot_xmin=0.0
            args.mktot_ymax=50.0
            args.mktot_ymin=0.0
            args.night='20180611'
            args.offset_exposure_indices=[1, 2, 3]
            args.optimal_ang_off_smoothing=0.05
            args.ra=13.8447
            args.reduction_dir='reductions'
            args.remove_tmp=True
            args.shifts_dir='/Users/mxhf/work/MPE/hetdex/src/vdrp_rewrite/shifts'
            args.shotid='017'
            args.shuffle_cfg='config/shuffle.cfg'
            args.task='all'
            args.track=1
            args.use_tmp=False
            args.wfs1_magadd=5.0
            args.wfs2_magadd=5.0
            args.dither_offsets=[(0.,0.),(1.270,-0.730),(1.270,0.730)]

            main(args)

            print("Checking *.coo ...")
            self.cmp_test_files("20180611v017", "*.coo")

            print("Checking *.ap ...")
            self.cmp_test_files("20180611v017", "*.ap")

            print("Checking *.als ...")
            self.cmp_test_files("20180611v017", "*.als")

            print("Checking all.mch ...")
            self.cmp_test_files("20180611v017", "all.mch")

            print("Checking all.raw ...")
            self.cmp_test_files("20180611v017", "all.raw")

            print("Checking shout.ifustars ...")
            self.cmp_test_files("20180611v017", "shout.ifustars")

            print("Checking tmp_*.csv ...")
            self.cmp_test_files("20180611v017", "tmp_*.csv")

            print("Checking norm.dat ...")
            self.cmp_test_files("20180611v017", "norm.dat")

            print("Checking getoff_*.out ...")
            self.cmp_test_files("20180611v017", "getoff_*.out")

            print("Checking getoff2_*.out ...")
            self.cmp_test_files("20180611v017", "getoff2_*.out")

            print("Checking radec2_*.dat ...")
            self.cmp_test_files("20180611v017", "radec2_*.dat")

            print("Checking radec2_final.dat ...")
            self.cmp_test_files("20180611v017", "radec2_final.dat")

#            ff = ["radec2_exp01.dat", "radec2_exp01.dat", "radec2_exp01.dat", "radec2_final.dat"]
#            for f in ff:
#                print("Checking {}".format(f))
#                ra,dec,pa = read_radec("20180611v017/{}".format(f))
#                ra_fid,dec_fid,pa_fid = read_radec("test_fiducial/20180611v017/{}".format(f))
#                self.assertAlmostEqual(ra, ra_fid)
#                self.assertAlmostEqual(dec, dec_fid)
#                self.assertAlmostEqual(pa, pa_fid)

            print("Checking xy*.dat ...")
            self.cmp_test_files("20180611v017", "xy*.dat")

            print("Checking dithall.use ...")
            self.cmp_test_files("20180611v017", "dithall.use")



