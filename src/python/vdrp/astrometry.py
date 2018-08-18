#!/usr/bin/env python
""" Astrometry routine

Module to add astrometry to HETDEX catalgoues and images
Contains python translation of Karl Gebhardt

.. moduleauthor:: Maximilian Fabricius <mxhf@mpe.mpg.de>
"""
from __future__ import print_function
from matplotlib import pyplot as plt

from numpy import loadtxt
import argparse
import os
import glob
import shutil
import sys
import ConfigParser
import logging
import subprocess
from astropy.io import fits
from astropy.io import ascii
import tempfile
import numpy as np
from collections import OrderedDict

#import scipy
from scipy.interpolate import UnivariateSpline

from distutils import dir_util

import path
from astropy import table
from astropy.table import Table

from astropy.stats import biweight_location as biwgt_loc
from astropy.io import fits
from astropy.table import vstack


from pyhetdex.het import fplane
from pyhetdex.coordinates.tangent_projection import TangentPlane
import pyhetdex.tools.read_catalogues as rc
from pyhetdex import coordinates
from pyhetdex.coordinates import astrometry as phastrom


from vdrp.cofes_vis import cofes_4x4_plots
from vdrp import daophot
from vdrp import cltools
from vdrp import utils
from vdrp.daophot import DAOPHOT_ALS
from vdrp import utils
from vdrp.utils import read_radec, write_radec



def parseArgs():
    """ Parses configuration file and command line arguments.
    Command line arguments overwrite configuration file settiongs which
    in turn overwrite default values.

    Args:
        args (argparse.Namespace): Return the populated namespace.
    """

    # Do argv default this way, as doing it in the functional
    # declaration sets it at compile time.
    argv=None
    if argv is None:
        argv = sys.argv

    # Parse any conf_file specification
    # We make this parser with add_help=False so that
    # it doesn't parse -h and print help.
    conf_parser = argparse.ArgumentParser(
            description=__doc__, # printed with -h/--help
            # Don't mess with format of description
            formatter_class=argparse.RawDescriptionHelpFormatter,
            # Turn off help, so we print all options in response to -h
            add_help=False
        )
    conf_parser.add_argument("-c", "--conf_file",
            help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()

    defaults = {}
    defaults["use_tmp"]  = "False"
    defaults["remove_tmp"]  = "True"
    defaults["logfile"] = "astrometry.log"
    defaults["reduction_dir"] = "/work/04287/mxhf/maverick/red1/reductions/"
    defaults["cofes_vis_vmin"] = -15.
    defaults["cofes_vis_vmax"] = 25.
    defaults["daophot_sigma"]  = 2
    defaults["daophot_xmin"] = 4
    defaults["daophot_xmax"] = 45
    defaults["daophot_ymin"] = 4
    defaults["daophot_ymix"] = 45
    defaults["daophot_opt"] = "vdrp/config/daophot.opt"
    defaults["daophot_phot_psf"]    = "vdrp/config/use.psf"
    defaults["daophot_photo_opt"]   = "vdrp/config/photo.opt"
    defaults["daophot_allstar_opt"] = "vdrp/config/allstar.opt"
    defaults["mktot_ifu_grid"] = "vdrp/config/ifu_grid.txt"
    defaults["mktot_magmin"] = 0.
    defaults["mktot_magmax"] = 21.
    defaults["mktot_xmin"] = 0.
    defaults["mktot_xmax"] = 50.
    defaults["mktot_ymin"] = 0.
    defaults["mktot_ymax"] = 50.
    defaults["fluxnorm_mag_max"] = 19.
    defaults["fplane_txt"] = "vdrp/config/fplane.txt"
    defaults["shuffle_cfg"] = "vdrp/config/shuffle.cfg"
    defaults["acam_magadd"] = 5.
    defaults["wfs1_magadd"] = 5.
    defaults["wfs2_magadd"] = 5.
    defaults["add_radec_angoff"] = 0.1
    defaults["add_radec_angoff_trial"] = 0.1
    defaults["add_radec_angoff_trial_dir"] = "add_radec_angoff_trial"
    defaults["getoff2_radii"] = 11.,5.,3.
    defaults["mkmosaic_angoff"] = 1.8
    defaults["task"] = "all"
    defaults["offset_exposure_indices"] = "1,2,3"
    defaults["optimal_ang_off_smoothing"] = 0.05

    if args.conf_file:
        config = ConfigParser.SafeConfigParser()
        config.read([args.conf_file])
        defaults.update(dict(config.items("Astrometry")))

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h
    parser = argparse.ArgumentParser(
    # Inherit options from config_parser
    parents=[conf_parser]
    )
    parser.set_defaults(**defaults)
    parser.add_argument("--logfile", type=str, help="Filename for log file.")
    parser.add_argument("--use_tmp", type=str, help="Use a temporary directory. Result files will be copied to NIGHTvSHOT.")
    parser.add_argument("--remove_tmp", type=str, help="Remove temporary directory after completion.")
    parser.add_argument("--reduction_dir", type=str, help="Directory that holds panacea reductions. Subdriectories with name like NIGHTvSHOT must exist.")
    parser.add_argument("--cofes_vis_vmin", type=float, help="Minimum value (= white) for matrix overview plot.")
    parser.add_argument("--cofes_vis_vmax", type=float, help="Maximum value (= black) for matrix overview plot.")
    parser.add_argument("--daophot_sigma",  type=float, help="Daphot sigma value.")
    parser.add_argument("--daophot_xmin",  type=float, help="X limit for daophot detections.")
    parser.add_argument("--daophot_xmax",  type=float, help="X limit for daophot detections.")
    parser.add_argument("--daophot_ymin",  type=float, help="Y limit for daophot detections.")
    parser.add_argument("--daophot_ymix",  type=float, help="Y limit for daophot detections.")
    parser.add_argument("--daophot_phot_psf",  type=str, help="Filename for daophot PSF model.")
    parser.add_argument("--daophot_opt",  type=str, help="Filename for daophot configuration.")
    parser.add_argument("--daophot_photo_opt",  type=str, help="Filename for daophot photo task configuration.")
    parser.add_argument("--daophot_allstar_opt",  type=str, help="Filename for daophot allstar task configuration.")
    parser.add_argument("--mktot_ifu_grid",  type=str, help="Name of file that holds gird of IFUs offset fit (mktot).")
    parser.add_argument("--mktot_magmin",  type=float, help="Magnitude limit for offset fit (mktot).")
    parser.add_argument("--mktot_magmax",  type=float, help="Magnitude limit for offset fit (mktot).")
    parser.add_argument("--mktot_xmin",  type=float, help="X limit for offset fit (mktot).")
    parser.add_argument("--mktot_xmax",  type=float, help="X limit for offset fit (mktot).")
    parser.add_argument("--mktot_ymin",  type=float, help="Y limit for offset fit (mktot).")
    parser.add_argument("--mktot_ymax",  type=float, help="Y limit for offset fit (mktot).")
    parser.add_argument("--fluxnorm_mag_max",  type=float, help="Magnitude limit for flux normalisation (mktot).")
    parser.add_argument("--fplane_txt",  type=str, help="Filename for fplane file.")
    parser.add_argument("--shuffle_cfg",  type=str, help="Filename for shuffle configuration.")
    parser.add_argument("--acam_magadd",  type=float, help="do_shuffle acam magadd.")
    parser.add_argument("--wfs1_magadd",  type=float, help="do_shuffle wfs1 magadd.")
    parser.add_argument("--wfs2_magadd",  type=float, help="do_shuffle wfs2 magadd.")
    parser.add_argument("--add_radec_angoff",  type=float, help="Angular offset to add during conversion of x/y coordinate to RA/Dec.")
    parser.add_argument("--add_radec_angoff_trial",  type=str, help="Trial values for angular offsets.")
    parser.add_argument("--add_radec_angoff_trial_dir",  type=str, help="Directory to save results of angular offset trials.")
    parser.add_argument('--getoff2_radii', type=str, help="Comma separated list of matching radii for astrometric offset measurement.")
    parser.add_argument("--mkmosaic_angoff",  type=float, help="Angular offset to add for creation of mosaic image.")
    parser.add_argument("-t", "--task",  type=str, help="Task to execute.")
    parser.add_argument("--offset_exposure_indices",  type=str)
    parser.add_argument("--optimal_ang_off_smoothing", type=float, help="Smothing value for smoothing spline use for measurement of optimal angular offset value.")

    # positional arguments
    parser.add_argument('night', metavar='night', type=str,
            help='Night of observation (e.g. 20180611).')
    parser.add_argument('shotid', metavar='shotid', type=str,
            help='Shot ID (e.g. 017).')
    parser.add_argument('ra', metavar='ra', type=float,
            help='RA of the target in decimal hours.')
    parser.add_argument('dec', metavar='dec', type=float,
            help='Dec of the target in decimal hours degree.')
    parser.add_argument('track', metavar='track', type=int, choices=[0, 1],
            help='Type of track: 0: East 1: West')

    args = parser.parse_args(remaining_argv)


    # shoul in principle be able to do this with accumulate???
    args.use_tmp = args.use_tmp == "True"
    args.remove_tmp = args.remove_tmp == "True"
    args.getoff2_radii = [float(t) for t in args.getoff2_radii.split(",")]
    args.add_radec_angoff_trial = [float(offset) for offset in  args.add_radec_angoff_trial.split(",")]

    args.offset_exposure_indices = [int(t) for t in args.offset_exposure_indices.split(",")]

    return args


def cp_post_stamps(args, wdir):
    """ Copy CoFeS (collapsed IFU images). 

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.

    Raises:
        Exception
    """
    ## find the IFU postage stamp fits files and copy them over
    pattern = os.path.join( args.reduction_dir, "{}/virus/virus0000{}/*/*/CoFeS*".format( args.night, args.shotid ) )
    logging.info("Copy {} files to {}".format(pattern,wdir))
    cofes_files = glob.glob(pattern)
    if len(cofes_files) == 0:
        raise Exception("Found no postage stamp images. Please check your reduction_dir in config file.")
    already_warned = False
    for f in cofes_files:
        h,t = os.path.split(f)
        target_filename = t[5:20] + t[22:26] + ".fits"
        if os.path.exists(os.path.join(wdir,target_filename)):
            if not already_warned:
                logging.warning("{} already exists in {}, skipping, won't warn about other files....".format(target_filename,wdir))
                already_warned = True
            continue

        shutil.copy2(f, os.path.join(wdir,target_filename) )


def mk_post_stamp_matrix(args,  wdir, prefixes):
    """ Create the IFU postage stamp matrix image.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
        cofes_files (list): List of CoFeS file names (collapsed IFU images).
    """
    # create the IFU postage stamp matrix image
    logging.info("Creating the IFU postage stamp matrix images ...")
    exposures = np.unique([p[:15] for p in prefixes])

    with path.Path(wdir):
        for exp in exposures:
            outfile_name = exp + ".png"
            logging.info("Creating {} ...".format(outfile_name))
            cofes_4x4_plots(prefix = exp, outfile_name = outfile_name, vmin = args.cofes_vis_vmin, vmax = args.cofes_vis_vmax, logging=logging)


def daophot_find(args,  wdir, prefixes):
    """ Run initial daophot find.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
        prefixes (list): List file name prefixes for the collapsed IFU images.
    """
    logging.info("Running initial daophot find...")
    # Create configuration file for daophot.
    shutil.copy2(args.daophot_opt, os.path.join(wdir, "daophot.opt") )
    with path.Path(wdir):
        for prefix in prefixes:
            # execute daophot
            daophot.daophot_find(prefix, args.daophot_sigma,logging=logging)
            # filter ouput
            daophot.filter_daophot_out(prefix + ".coo", prefix + ".lst", args.daophot_xmin,args.daophot_xmax,args.daophot_ymin,args.daophot_ymix)


def daophot_phot_and_allstar(args, wdir, prefixes):
    """ Runs daophot photo and allstar on all IFU postage stamps.
    Produces *.ap and *.als files.
    Analogous to run4a.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
        prefixes (list): List file name prefixes for the collapsed IFU images.
    """
    # run initial daophot phot & allstar
    logging.info("Running daophot phot & allstar ...")
    # Copy configuration files for daophot and allstar.
    shutil.copy2(args.daophot_photo_opt, os.path.join(wdir, "photo.opt") )
    shutil.copy2(args.daophot_allstar_opt, os.path.join(wdir, "allstar.opt") )
    shutil.copy2(args.daophot_phot_psf, os.path.join(wdir, "use.psf"))
    with path.Path(wdir):
        for prefix in prefixes:
            # first need to shorten file names such
            # that daophot won't choke on them.
            daophot.daophot_phot(prefix,logging=logging)
            daophot.allstar(prefix,logging=logging)


def mktot(args, wdir, prefixes):
    """ Reads all *.als files. Put detections on a grid
    corresponding to the IFU position in the focal plane as defined in
    config/ifu_grid.txt (should later become fplane.txt.
    Then produces all.mch.

    Note:
        Analogous to run6 and run6b.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (string): Work directory.
        prefixes (list): List file name prefixes for the collapsed IFU images.
    """
    # read IFU grid definition file (needs to be replaced by fplane.txt)
    ifugird = Table.read(args.mktot_ifu_grid, format='ascii')

    with path.Path(wdir):
        exposures = np.unique([p[:15] for p in prefixes])

        for exp in exposures:
            fnout = exp + "tot.als"
            logging.info("Creating {}".format(fnout))

            with open(fnout, 'w') as fout:
                s  = " NL   NX   NY  LOWBAD HIGHBAD  THRESH     AP1  PH/ADU  RNOISE    FRAD\n"
                s += "  1   49   49  -113.6 84000.0    7.93    1.00    1.27    1.06    3.00\n"
                s += "\n"
                fout.write(s)

                count = 0
                for prefix in prefixes:
                    if not prefix.startswith(exp):
                        continue
                    ifuslot = int(prefix[-3:])

                    # find xoffset and yoffset for current IFU slot
                    jj = ifugird['IFUSLOT'] == ifuslot
                    if sum(jj) < 1:
                        logging.warning("mktot: IFU slot {} not found in {}.".format(ifuslot, args.mktot_ifu_grid))
                        continue
                    ifugird['X'][jj][0]
                    ifugird['Y'][jj][0]
                    xoff = ifugird['X'][jj][0]
                    yoff = ifugird['Y'][jj][0]

                    # read daophot als input file
                    try:
                        als = daophot.DAOPHOT_ALS.read(prefix + ".als")
                    except:
                        logging.warning("mktot: WARNING Unable to read " + prefix + ".als")
                        continue

                    # filter according to magnitude and x and y range
                    ii  = als.data['MAG'] > args.mktot_magmin
                    ii *= als.data['MAG'] < args.mktot_magmax
                    ii *= als.data['X']   > args.mktot_xmin
                    ii *= als.data['X']   < args.mktot_xmax
                    ii *= als.data['Y']   > args.mktot_ymin
                    ii *= als.data['Y']   < args.mktot_ymax

                    count += sum(ii)

                    for d in als.data[ii]:
                        #s = "{:03d} {:8.3f} {:8.3f} {:8.3f}\n".format( d['ID'], d['X']+xoff, d['Y']+yoff, d['MAG'] )
                        s = "{:d} {:8.3f} {:8.3f} {:8.3f}\n".format( d['ID'], d['X']+xoff, d['Y']+yoff, d['MAG'] )
                        fout.write(s)

                logging.info("{} stars in {}.".format(count, fnout))
        # produce all.mch like run6b
        with open("all.mch", 'w') as fout:
            dither_offsets = [(0.,0.),(1.270,-0.730),(1.270,0.730)]
            s = ""
            for i in range(len(exposures)):
                s  += " '{:30s}'     {:.3f}     {:.3f}   1.00000   0.00000   0.00000   1.00000     0.000    0.0000\n".format(exposures[i] + "tot.als",dither_offsets[i][0], dither_offsets[i][1])
            fout.write(s)


def rmaster(args, wdir):
    """ Executes daomaster. This registers the sets of detections
    for the thre different exposrues with respec to each other.

    Note:
        Analogous to run8b.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
    """
    logging.info("Running daomaster.")

    with path.Path(wdir):
        daophot.rm(["all.raw"])
        daophot.daomaster(logging=logging)


def getNorm(all_raw, mag_max):
    """ Comutes the actual normalisation for flux_norm.

    Note:
        Analogous to run9.

    Args:
        all_raw (str): Output file name of daomaster, usuall all.raw.
        mag_max (float): Magnitude cutoff for normalisation. Fainter objects will be ignored.
    """
    def mag2flux(m):
        return 10**((25-m)/2.5)

    ii  = all_raw[:,3] < mag_max
    ii *= all_raw[:,5] < mag_max
    ii *= all_raw[:,7] < mag_max

    f1 = mag2flux(all_raw[ii,3])
    f2 = mag2flux(all_raw[ii,5])
    f3 = mag2flux(all_raw[ii,7])

    favg = (f1+f2+f3)/3.
    return biwgt_loc(f1/favg),biwgt_loc(f2/favg),biwgt_loc(f3/favg)


def flux_norm(args, wdir, infile='all.raw', outfile='norm.dat'):
    """ Reads all.raw and compute relative flux normalisation
    for the three exposures.

    Note:
        Analogous to run9.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
        infile (str): Output file of daomaster.
        outfile (str): Filename for result file.
    """
    logging.info("Computing flux normalisation between exposures 1,2 and 3.")
    mag_max = args.fluxnorm_mag_max
    with path.Path(wdir):
        all_raw = loadtxt(infile, skiprows=3)
        n1,n2,n3 = getNorm(all_raw, mag_max )
        logging.info("flux_norm: Flux normalisation is {:10.8f} {:10.8f} {:10.8f}".format(n1,n2,n3) )
        with open(outfile, 'w') as f:
            s = "{:10.8f} {:10.8f} {:10.8f}".format(n1,n2,n3)
            f.write(s)


def redo_shuffle(args, wdir):
    """
    Reruns shuffle to obtain catalog of IFU stars.

    Creates a number of output files, most importantly
    `shout.ifustars` which is used as catalog for the offset computation.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
    """
    shutil.copy2(args.shuffle_cfg, wdir)
    shutil.copy2(args.fplane_txt, wdir)
    with path.Path(wdir):
        try:
            os.remove(shout.ifustars)
        except:
            pass

        RA0      = args.ra
        DEC0     = args.dec
        radius   = 0.
        track    = args.track
        ifuslot  = 0
        x_offset = 0.
        y_offset = 0

        daophot.rm(['shout.acamstars','shout.ifustars','shout.info','shout.probestars','shout.result'])
        logging.info("Rerunning shuffle for RA = {}, Dec = {} and track = {} ...".format(RA0, DEC0, track))
        cmd  = "do_shuffle -v --acam_magadd {:.2f} --wfs1_magadd {:.2f} --wfs2_magadd {:.2f}".format(args.acam_magadd, args.wfs1_magadd, args.wfs2_magadd)
        cmd += " {:.6f} {:.6f} {:.1f} {:d} {:d} {:.1f} {:.1f}".format(RA0, DEC0, radius, track, ifuslot, x_offset, y_offset )
        logging.info("redo_shuffle: Calling shuffle with {}.".format(cmd))
        subprocess.call(cmd, shell=True)


def get_ra_dec_orig(args,wdir):
    """
    Reads first of the many multi* file'd headers to get the RA, DEC, PA guess from the telescope.

    Notes:
        Creates radec.orig
    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
    """
    pattern = os.path.join( args.reduction_dir, "{}/virus/virus0000{}/*/*/multi_???_*LL*fits".format( args.night, args.shotid ) )
    multi_files = glob.glob(pattern)
    if len(multi_files) == 0:
        raise Exception("Found no multi file in {}. Please check reduction_dir in configuration file.".format(args.reduction_dir))
    h = fits.getheader(multi_files[0])
    ra0  = h["TRAJCRA"]
    dec0 = h["TRAJCDEC"]
    pa0  = h["PARANGLE"]
    logging.info("get_ra_dec_orig: Original RA,DEC,PA = {},{},{}".format(ra0, dec0, pa0))
    with path.Path(wdir):
        utils.write_radec(ra0, dec0, pa0, "radec.orig")


def add_ra_decOld(args, wdir, exp_prefixes, ra, dec, pa, radec_outfile='tmp.csv'):
    """
    Call add_ra_dec to compute for detections in IFU space the corresponding RA/DEC
    coordinates.


    Requires, fplane.txt, radec.orig.
    Creates primarely EXPOSURE_tmp.csv but also radec.dat.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
        exp_prefixes (list): List file name prefixes for the collapsed IFU images for one exposure (typically the first).
        ra (float): Focal plane center RA.
        dec (float): Focal plane center Dec.
        pa (float): Positions angle.
        radec_outfile (str): Filename that will contain output from add_ra_dec (gets overwritten!).

    """
    with path.Path(wdir):
        fp = fplane.FPlane("fplane.txt")

        # collect als files for all IFUs that are contained in the fplane file. 
        als_files = []
        for prefix in exp_prefixes:
            ifuslot = prefix[-3:]
            if not ifuslot in fp.ifuslots:
                logging.warning("IFU slot {} not contained in fplane.txt.".format(ifuslot))
                continue
            fn = prefix + ".als"
            als_files.append(fn)

        daophot.rm([radec_outfile])
        cmd = 'add_ra_dec --fplane fplane.txt --fout {} --ftype daophot_allstar --astrometry {} {} {} --ihmp-regex "(\d\d\d)(\.)" '.format(radec_outfile, ra, dec, pa)
        logging.info("Calling {} ".format(cmd))
        for f in als_files:
            logging.info("        {}".format(f))
            cmd += " {}".format(f)
        subprocess.call(cmd, shell=True)



def get_als_files(fp, exp_prefixes):
    """
    Derives for a list of exposure prefixes a list
    of *.als files, but rejects any that refer to an IFU slot
    which is not contained in the fplane.

    Args:
        exp_prefixes (list): List of epxosure prefixes.

    Returns:
        (list): List of *.als files.
    """
    # collect als files for all IFUs that are contained in the fplane file. 
    als_files = []
    for prefix in exp_prefixes:
        ifuslot = prefix[-3:]
        if not ifuslot in fp.ifuslots:
            logging.warning("IFU slot {} not contained in fplane.txt.".format(ifuslot))
            continue
        fn = prefix + ".als"
        als_files.append(fn)
    return als_files


def load_als_data(als_files):
    """ Load set of als files.
    Args:
        als_files (list): List of file names.

    Returns:
        (OrderedDict):  Dictionary with als data for each IFU slot.
    """
    # work out the IFU slot from the file name
    als_data = OrderedDict()
    for fn in als_files:
        ihmp = fn[-7:-4]
        data = rc.read_daophot(fn)
        als_data[ihmp] = data
    return als_data


def add_ra_dec(args, wdir, als_data, ra, dec, pa, fp, radec_outfile='tmp.csv'):
    """
    Call add_ra_dec to compute for detections in IFU space the corresponding RA/DEC
    coordinates.

    New version, direct python call to pyheted.coordinates.tangent_projection.

    Requires, fplane.txt, radec.orig.
    Creates primarely EXPOSURE_tmp.csv but also radec.dat.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
        als_data (dict): Dictionary with als data for each IFU slot.
        ra (float): Focal plane center RA.
        dec (float): Focal plane center Dec.
        pa (float): Positions angle.
        fp (FPlane): Focal plane object.
        radec_outfile (str): Filename that will contain output from add_ra_dec (gets overwritten!).

    """
    with path.Path(wdir):
        fp = fplane.FPlane("fplane.txt")

        # Carry out required changes to astrometry
        rot = 360.0 - (pa + 90.)
        # Set up astrometry from user supplied options
        tp = TangentPlane(ra, dec, rot)

        # Loop over the files, adding ra, dec
        tables = []
        for ihmp in als_data:
            x, y, table = als_data[ihmp]

            # skip empty tables
            if len(x) < 1:
                continue

            ifu = fp.by_ifuslot(ihmp)

            # remember to flip x,y
            xfp = x + ifu.y
            yfp = y + ifu.x

            ra, dec = tp.xy2raDec(xfp, yfp)

            table['ra'] = ra
            table['dec'] = dec
            table['ifuslot'] = ihmp
            table['xfplane'] = xfp
            table['yfplane'] = yfp

            tables.append(table)

        # output the combined table
        table_out = vstack(tables)
        logging.info("Writing output to {:s}".format(radec_outfile))
        table_out.write(radec_outfile, comment='#',overwrite=True)


def compute_optimal_ang_off(wdir, smoothing=0.05, PLOT=True):
    """
    Computes the optimal angular offset angle by findin the minimal
    RMS of a set of different trial angles.

    Takes (if exist) all three different exposures into account and computes
    weighted average ange (weighted by number of stars that went into the fit).

    The RMS(ang_off) are interpolate with a smoothing spline. The smoothing value
    is a parameter to this function.

    Args:
        wdir (string): Directory that holds the angular offset trials (e.g. 20180611v017/add_radec_angoff_trial)
    Returns:
        (float): Optimal offset angle.
    """
    colors = ['red','green','blue']
    exposures = ['exp01','exp02', 'exp03']

    # load getoff2 data for all exposures
    results = Table(names=['exposure','ang_off','nstar','RMS'], dtype=['S5',float, int, float])
    for exp in exposures:
        list = glob.glob("{}/getoff2_{}*Deg.out".format(wdir,exp))
        if len(list) == 0:
            logging.warning("Found no files for exposure {}".format(exp))
            continue

        for filename in list:
            # count how many stars contribute
            with open(filename.replace('getoff2','getoff')) as f:
                ll = f.readlines()
            nstar = len(ll)
            # now load the getoff2 to read the RMS
            ang = float( filename.replace('Deg.out','').split("_")[-1] )
            with open(filename) as f:
                ll = f.readlines()
            try:
                tt = ll[0].split()
                rms_dra = float( tt[2] )
                rms_ddec = float( tt[3] )
                results.add_row( [ exp, ang, nstar, np.sqrt(rms_dra**2. + rms_ddec**2.)]  )
            except:
                logging.error("Parsing {}".format(filename))

    if len(results) == 0:
        logging.error("Found no data for angular offset trials.")
        return np.nan

    if PLOT:
        fig = plt.figure(figsize = [7,7])
        ax = plt.subplot(111)

    # angular subgrid for interpolation
    aa = np.arange( results['ang_off'].min(), results['ang_off'].max() , 0.01)
    aamin = Table(names=["exposure", "ang_off_min", "nstar_min", "RMS_min"], dtype=['S5',float, int, float])
    # iterate over all 1-3 exposures.
    for i,exp in enumerate(exposures):
        ii = results['exposure'] == exp
        x = results['ang_off'][ii]
        y = results['RMS'][ii]
        n = results['nstar'][ii]
        jj = np.argsort(x)
        x = x[jj]
        y = y[jj]
        n = n[jj]

        # old cubic interpolation
        #f = interpolate.interp1d(x, y, kind='cubic', bounds_error=False)
        f = UnivariateSpline(x, y, s=smoothing)
        # this is a bit silly, but since the number of stars may change as a function of
        # angle we also need to interpolate those.
        fn = UnivariateSpline(x, n, s=smoothing)
        # find best offset angle
        imin = np.nanargmin(f(aa))
        amin = aa[imin]
        rms_min = f(aa[imin])
        nstar_min = fn(aa[imin])
        aamin.add_row([exp, amin, nstar_min, rms_min] )

        if PLOT:
            plt.plot(x,y ,'o', c=colors[i], label=exp)
            plt.plot(aa,f(aa) ,'-', c=colors[i])
            plt.axvline(amin,color=colors[i])
            plt.text(amin,1.5,"{:.3f}Deg # stars = {:.1f}".format(amin, nstar_min, ), color=colors[i], rotation=90., ha='right')

    # average optimal offset angle accross all exposures
    ang_off_avg = np.sum( aamin['ang_off_min']*aamin['nstar_min'])/np.sum(aamin['nstar_min'])

    if PLOT:
        plt.axvline(ang_off_avg,color="k")
        plt.legend()
        plt.xlabel("Offset angle")
        plt.ylabel("RMS")
        plt.text(.1,.9, "avg. optimal\noffset angle\nis {:.5} Deg".format( ang_off_avg), transform=ax.transAxes)
        fig.tight_layout()
        plt.savefig(os.path.join(wdir, "ang_off.pdf"),overwrite=True)

    return ang_off_avg

def compute_offset(args, wdir, prefixes, final_ang_offset=None, shout_ifustars = 'shout.ifustars',
        NEW_ADD_RA_DEC=True):
    """
    Requires, fplane.txt, radec.orig.
    Creates primarely EXPOSURE_tmp.csv but also radec2.dat.

    Compute offset in RA DEC  by matching detected stars in IFUs
    against the shuffle profived RA DEC coordinates.
    Notes:
        Analogous to rastrom3.
        Creates radec.dat, radec2.dat and
        radec_TRIAL_OFFSET_ANGLE.dat, radec_TRIAL_OFFSET_ANGLE2.dat.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
        prefixes (list): List file name prefixes for the collapsed IFU images.
        final_ang_offset (float): Final angular offset to use. This overwrite the values in args.
        shout_ifustars (str): Shuffle output catalog of IFU stars.
    """
    shout_ifustars = 'shout.ifustars'

    def write_ra_dec_dats(ra, dec, pa, exp_index, angoff, ra_offset, dec_offset, nominal=False):
        sangoff = ""
        if not nominal:
            sangoff = '_{:06.3f}Deg'.format(angoff)

        # write results to radec_exp??.dat
        fnout = "radec_exp{:02d}{}.dat".format(exp_index,sangoff)
        utils.write_radec(ra*15., dec, pa + angoff, fnout)
        logging.info("Wrote {}".format(fnout))

        # write results to radec2_exp??.dat
        fnout = "radec2_exp{:02d}{}.dat".format(exp_index, sangoff)
        utils.write_radec(ra*15. + ra_offset, dec + dec_offset,pa + angoff,
                fnout)
        logging.info("Wrote {}".format(fnout))


    with path.Path(wdir):
        radii = args.getoff2_radii

        # Here we iterate over all angular offset angles
        # as configured in the config file, parameter add_radec_angoff_trial.
        # Should add_radec_angoff not be in that list, we add it here
        # and the third line makes sure that the nominal angle
        # is the last one that we compute. This is important
        # such that all the correct output files are in place for the downstream 
        # functions.
        angoffsets         = args.add_radec_angoff_trial
        nominal_angoffset  = args.add_radec_angoff
        if final_ang_offset != None:
            logging.info("compute_offset: Using final angular offset value of {} Deg.".format(final_ang_offset))
            angoffsets = []
            nominal_angoffset = final_ang_offset
        angoffsets = filter( lambda x : x != nominal_angoffset, angoffsets) + [nominal_angoffset]

        # Give comprehensive information about the iterations.
        s = ""
        for r in radii:
            s+= "{}\" ".format(r)
        logging.info("compute_offset: Computing offsets with using following sequence of matching radii: {}".format(s))
        logging.info("compute_offset:  Using nominal angular offset value of {} Deg. ".format(args.add_radec_angoff))
        s = ""
        for a in args.add_radec_angoff_trial:
            s+= "{} Deg ".format(a)
        logging.info("compute_offset:  Also computing offsets for the following set of trial angles: {}".format(s) )

        # will contain results of angular offset trials
        utils.createDir(args.add_radec_angoff_trial_dir)


        for angoff in angoffsets:
            # collect the prefixes that belong to the first exposure
            # for now only do first exposure, later can do them all
            exposures = np.sort( np.unique([p[:15] for p in prefixes]) )

            # loop over all exposures in configuration file
            for exp_index in args.offset_exposure_indices:
                if exp_index > len(exposures):
                    logging.warning("compute_offset: Have no data for exposure {}. Skipping ...".format(exp_index))
                    continue
                exp = exposures[exp_index-1] # select first exposure
                exp_prefixes = []
                # collect all als files for this exposure
                for prefix in prefixes:
                    if not prefix.startswith(exp):
                        continue
                    exp_prefixes.append(prefix)


                # Convert radec.orig to radec.dat, convert RA to degress and add angular offset
                # mF: Not sure if we will need radec.dat later, creating it for now.
                ra,dec,pa = utils.read_radec("radec.orig")

                # Now compute offsets iteratively with increasingly smaller matching radii.
                # Matching radii are defined in config file.
                ra_offset, dec_offset = 0., 0.
                for i, radius in enumerate(radii):
                    logging.info("compute_offset: Angular offset {:.3} Deg, getoff2 iteration {}, matching radius = {}\"".format(angoff, i+1, radius))
                    radec_outfile='tmp_exp{:02d}.csv'.format(exp_index)
                    logging.info("compute_offset: Adding RA & Dec to detections, applying offsets ra_offset,dec_offset,pa_offset = {},{},{}".format( ra_offset, dec_offset, angoff) )
                    # Call add_ra_dec, add offsets first.
                    new_ra, new_dec, new_pa = ra * 15. + ra_offset, dec + dec_offset, pa + angoff
                    if NEW_ADD_RA_DEC:
                        # New direct call to pyhetdex
                        # preload the als data.
                        fp = fplane.FPlane("fplane.txt")
                        als_files = get_als_files(fp, exp_prefixes)
                        als_data = load_als_data(als_files)
                        add_ra_dec(args, wdir, als_data, ra=new_ra, dec=new_dec,\
                            pa=new_pa, fp=fp, radec_outfile=radec_outfile)
                        add_ra_dec(args, wdir, als_data, ra=new_ra, dec=new_dec, pa=new_pa, fp=fp, radec_outfile=radec_outfile)
                    else:
                        # Old command line call
                        add_ra_decOld(args, wdir, exp_prefixes, ra=new_ra, dec=new_dec, pa=new_pa, radec_outfile=radec_outfile)

                    # Now compute offsets.
                    logging.info("compute_offset: Computing offsets ...")
                    dra_offset, ddec_offset = cltools.getoff2(radec_outfile, shout_ifustars, radius, ra_offset=0., dec_offset=0., logging=logging)
                    ra_offset, dec_offset =  ra_offset+dra_offset, dec_offset+ddec_offset
                    logging.info("compute_offset: End getoff2 iteration {}: Offset adjusted by {:.6f}, {:.6f} to {:.6f}, {:.6f}".format(i+1, dra_offset, ddec_offset, ra_offset, dec_offset))
                    logging.info("compute_offset: ")
                    logging.info("compute_offset: ")

                # Copy getoff.out and getoff2.out to args.add_radec_angoff_trial_dir
                sangoff = '_{:06.3f}Deg'.format(angoff)
                fnout = os.path.join( args.add_radec_angoff_trial_dir, "getoff_exp{:02d}{}.out".format(exp_index,sangoff) )
                shutil.copy2("getoff.out", fnout)
                fnout = os.path.join( args.add_radec_angoff_trial_dir, "getoff2_exp{:02d}{}.out".format(exp_index,sangoff) )
                shutil.copy2("getoff2.out", fnout)

                shutil.move("getoff.out", "getoff_exp{:02d}.out".format(exp_index))
                shutil.move("getoff2.out", "getoff2_exp{:02d}.out".format(exp_index))
                # Write radec_XXXDeg.dat and radec2_XXXDeg.dat
                with path.Path(args.add_radec_angoff_trial_dir):
                    write_ra_dec_dats(ra, dec, pa, exp_index, angoff, ra_offset, dec_offset, nominal=False)
                # if the current offset angle is the nominal one, then also write
                # radec.dat and radec2.dat witouh angle information in filename.
                if angoff == args.add_radec_angoff:
                    write_ra_dec_dats(ra, dec, pa, exp_index, angoff, ra_offset, dec_offset, nominal=True)


def add_ifu_xyOld(args, wdir):
    """ Adds IFU x y information to stars used for matching,
    and save to xy_expNN.dat.
    Requires: getoff.out, radec2.dat
    Analogous to rastrom3.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
    """
    logging.info("Creating xy_expNN.dat...")
    with path.Path(wdir):
        # loop over all exposures in configuration file
        for exp_index in args.offset_exposure_indices:
            fngetoff_out = "getoff_exp{:02d}.out".format(exp_index)
            if not os.path.exists(fngetoff_out):
                logging.warning("Have no {} for exposure {}. Check your configuration (offset_exposure_indices). Skipping ...".format(fngetoff_out, exp_index))
                continue

            # read ra dec postions of reference stars and detections
            # from getoff.out
            # Produce new tables that add_ifu_xy understands.
            t = Table.read(fngetoff_out, format="ascii.fast_no_header")
            t1 = Table([t['col3'], t['col4'], t['col7']], names=["RA","DEC","IFUSLOT"])
            t2 = Table([t['col5'], t['col6'], t['col7']], names=["RA","DEC","IFUSLOT"])
            t1.write("t1.csv", format="ascii.fast_csv", overwrite=True)
            t2.write("t2.csv", format="ascii.fast_csv", overwrite=True)

            # read ra,dec, pa from radec2.dat
            ra,dec,pa = utils.read_radec("radec2_exp{:02d}.dat".format(exp_index))

            daophot.rm(["t3.csv","t4.csv"])
            cmd = "add_ifu_xy --fplane fplane.txt --ra-name RA --dec-name DEC --astrometry {:.6f} {:.6f} {:.6f} t1.csv t3.csv".format(ra,dec,pa)
            subprocess.call(cmd, shell=True)
            cmd = "add_ifu_xy --fplane fplane.txt --ra-name RA --dec-name DEC --astrometry {:.6f} {:.6f} {:.6f} t2.csv t4.csv".format(ra,dec,pa)
            subprocess.call(cmd, shell=True)

            t3 = Table.read("t3.csv", format="ascii.fast_csv")
            t4 = Table.read("t4.csv", format="ascii.fast_csv")

            for c in t3.columns:
                t3.columns[c].name = t3.columns[c].name + "1"
            for c in t4.columns:
                t4.columns[c].name = t4.columns[c].name + "2"

            t = table.hstack([t3,t4])
            t.write('xy_exp{:02d}.dat'.format(exp_index), format="ascii.fixed_width", delimiter='', overwrite=True)
            # this would be analogous to Karl's format
            #t.write('xy.dat', format="ascii.fast_no_header"


def combine_radec(wdir, PLOT=True):
    """
    Computes - based on the RA Dec information of the individual exposures
    (from radec2_exp0?.dat) the final RA/Dec for the shot.

    Notes:
        Creates radec2_final.dat.
        Optionally create a plot indicating the individual exposure positions.

    Args:
        wdir (str): Work directory.
    """
    logging.info("combine_radec: Combining RA, Dec positions of all exposures to final shot RA, Dec.")
    dither_offsets = [(0.,0.),(1.270,-0.730),(1.270,0.730)]
    ff = np.sort( glob.glob(wdir + "/radec2_exp??.dat") )
    ra0,dec0,pa0 = read_radec(ff[0])
    translated = []
    if PLOT:
        fig = plt.figure(figsize=[7,7])
        ax = plt.subplot(111)
    for i,(f, offset) in enumerate(zip(ff, dither_offsets)):
        ra,dec,pa = read_radec(f)
        logging.info("combine_radec: Exposure {:d} RA,Dec = {:.6f},{:.6f}".format(i+1,ra,dec))
        rot = 360.0 - (pa + 90.)
        tp = TangentPlane(ra, dec, rot)

        xfp = 0.
        yfp = 0.
        _ra, _dec = tp.xy2raDec(xfp, yfp)

        if PLOT:
            plt.plot(_ra,_dec,'s',color='grey')
            plt.text(_ra+5e-6,_dec+5e-6,i+1)

        xfp = -offset[0]
        yfp = -offset[1]
        _ra, _dec = tp.xy2raDec(xfp, yfp)

        if PLOT:
            plt.plot(_ra,_dec,'o',color='grey')
            plt.text(_ra+5e-6,_dec+5e-6,i+1)

        translated.append([_ra, _dec])

    translated = np.array(translated)
    final_ra, final_dec = np.mean(translated[:,0]), np.mean(translated[:,1])
    dfinal_ra = np.std(translated[:,0])/np.cos(np.deg2rad(dec0))
    ddfinal_dec = np.std(translated[:,1])

    s1  = "RA = {:.6f} Deg +/- {:.3f}\"".format(final_ra, dfinal_ra*3600.)
    logging.info("combine_radec: Final shot  " + s1)
    s2  = "Dec = {:.6f} Deg +/- {:.3f}\" ".format(final_dec, ddfinal_dec*3600.)
    logging.info("combine_radec: Final shot  " + s2)
    if PLOT:
        plt.plot([],[],'s',color='grey',label="exposure center")
        plt.plot([],[],'o',color='grey',label="inferred shot center")
        l = plt.legend()
        plt.plot( [final_ra], [final_dec],'ko',markersize=10)
        plt.text(final_ra, final_dec,s1 + "\n" + s2,ha='right')
        plt.xlabel("RA [Deg]")
        plt.ylabel("Dec [Deg]")

    write_radec(final_ra,final_dec,pa0, os.path.join(wdir, "radec2_all.dat") )
    fig.tight_layout()
    plt.savefig(os.path.join(wdir, "radec2_all.pdf"))


def add_ifu_xy(args, wdir):
    """ Adds IFU x y information to stars used for matching,
    and save to xy_expNN.dat.
    Requires: getoff.out, radec2.dat
    Analogous to rastrom3.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
    """
    def ra_dec_to_xy(table_in, ra, dec, fp, tp):
        """ Little helper function that convinently wraps the call to
        pyhetdex.coordinates.astrometry.ra_dec_to_xy
        """
        ra = table_in["RA"]
        dec = table_in["DEC"]
        # find positions
        table_out = phastrom.ra_dec_to_xy(ra, dec, fp, tp)
        table_final = table.hstack([table_in, table_out])
        return table_final

    with path.Path(wdir):
        # loop over all exposures in configuration file
        for exp_index in args.offset_exposure_indices:
            logging.info("Creating xy_exp{:02d}.dat".format(exp_index) )

            fngetoff_out = "getoff_exp{:02d}.out".format(exp_index)
            if not os.path.exists(fngetoff_out):
                logging.warning("Have no {} for exposure {}. Check your configuration (offset_exposure_indices). Skipping ...".format(fngetoff_out, exp_index))
                continue

            t = Table.read(fngetoff_out, format="ascii.fast_no_header")
            t_detect_coor  = Table([t['col3'], t['col4'], t['col7']], names=["RA","DEC","IFUSLOT"])
            t_catalog_coor = Table([t['col5'], t['col6'], t['col7']], names=["RA","DEC","IFUSLOT"])

            # read ra,dec, pa from radec2.dat
            ra,dec,pa = utils.read_radec("radec2_exp{:02d}.dat".format(exp_index))

            # set up astrometry
            fp = fplane.FPlane("fplane.txt")
            # Carry out required changes to astrometry
            rot = 360.0 - (pa + 90.)
            # Set up astrometry from user supplied options
            tp = TangentPlane(ra, dec, rot)

            t_detect_coor_xy = ra_dec_to_xy(t_detect_coor, ra, dec, fp, tp)
            t_catalog_coor_xy = ra_dec_to_xy(t_catalog_coor, ra, dec, fp, tp)

            for c in t_detect_coor_xy.columns:
                #t_detect_coor_xy.columns[c].name = t_detect_coor_xy.columns[c].name + "1"
                t_detect_coor_xy.columns[c].name = \
                    t_detect_coor_xy.columns[c].name + "_det"
            for c in t_catalog_coor_xy.columns:
                #t_catalog_coor_xy.columns[c].name = t_catalog_coor_xy.columns[c].name + "2"
                t_catalog_coor_xy.columns[c].name = \
                    t_catalog_coor_xy.columns[c].name + "_cat"

            t = table.hstack([t_detect_coor_xy,t_catalog_coor_xy])
            t.write('xy_exp{:02d}.dat'.format(exp_index), format="ascii.fixed_width", delimiter='', overwrite=True)


def mkmosaic(args, wdir, prefixes):
    """Creates mosaic fits image.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
        prefixes (list): List file name prefixes for the collapsed IFU images.
    """
    with path.Path(wdir):
        logging.info("mkmosaic: Creating mosaic image.")
        # build mosaic from IFU images
        exposures = np.unique([p[:15] for p in prefixes])
        exp1 = exposures[0]
        # collect all als files for the first exposure
        pp = filter(lambda x : x.startswith(exp1), prefixes)
        logging.info("mkmosaic: Calling immosaicv ....")
        daophot.rm(['immosaic.fits'])
        cltools.immosaicv(pp, fplane_file = "fplane.txt", logging=logging)

        # rotate mosaic to correct PA on sky
        ra,dec,pa = utils.read_radec('radec2_exp01.dat')
        alpha = 360. - (pa + 90. + args.mkmosaic_angoff)

        logging.info("mkmosaic: Calling imrot with angle {} (can take a minute) ....".format(alpha))
        daophot.rm(['imrot.fits'])
        cltools.imrot("immosaic.fits", alpha, logging=logging)
        hdu = fits.open("imrot.fits")

        h = hdu[0].header
        h["CRVAL1"]  = ra
        h["CRVAL2"]  = dec
        h["CTYPE1"]  = "RA---TAN"
        h["CTYPE2"]  = "DEC--TAN"
        h["CD1_1"]   = -0.0002777
        h["CD1_2"]   = 0.
        h["CD2_2"]   = 0.0002777
        h["CD2_1"]   = 0
        h["CRPIX1"]  = 650.0
        h["CRPIX2"]  = 650.0
        h["CUNIT1"]  = "deg"
        h["CUNIT2"]  = "deg"
        h["EQUINOX"] = 2000

        hdu.writeto("{}v{}fp.fits".format(args.night, args.shotid),overwrite=True)


def project_xy(wdir, radec_file, fplane_file, ra, dec):
    """Translate *all* catalog stars to x/y to display then and to
    see which ones got matched.
    Call pyhetdex tangent_plane's functionality to project
    ra,dec to x,y.

    Args:
        wdir (str): Work directory.
        radec_file (str): File that contains shot ra dec position.
        fplane_file (str): Focal plane file filename.
        ra (list): List of ra positions (in float, degree).
        dec (list): List of dec positions (in float, degree).
    """
    # read ra,dec, pa from radec2.dat
    ra0,dec0,pa0 = utils.read_radec( os.path.join(wdir, radec_file))
    # Carry out required changes to astrometry
    rot = 360.0 - (pa0 + 90.)
    # Set up astrometry from user supplied options
    tp = phastrom.TangentPlane(ra0, dec0, rot)
    # set up astrometry
    fp = fplane.FPlane( os.path.join(wdir, fplane_file) )
    # find positions
    ifu_xy = phastrom.ra_dec_to_xy(ra, dec, fp, tp)
    return ifu_xy


def mk_match_matrix(wdir, ax, exp, image_files, fplane_file, shout_ifu_file, xy_file, radec_file):
    """ Creates the actual match plot for a specific exposures.
    This is a subroutine to mk_match_plots.

    Args:
        wdir (str): Work directory.
        prefixes (list): List file name prefixes for the collapsed IFU images.
        ax (pyplot.axes): Axes object to plot into.
        exp (str): Exposure string (e.g. exp01)
        image_files (list): List of file names.
        fplane_file (str): Focal plane file filename.
        shout_ifu_file (str): Shuffle IFU star catalog output filename.
        xy_file (str): Filename for list of matched stars, aka xy_exp??.dat.
        radec_file (str): File that contains shot ra dec position.
    """
    cmap = plt.cm.bone

    N = 1.
    tin = Table.read(os.path.join(wdir, shout_ifu_file), format='ascii')
    tout = Table( [tin['col2'], tin['col3'], tin['col4']], names=['id','ra','dec'])

    # load images
    images = OrderedDict()
    headers = OrderedDict()
    with path.Path(wdir):
        for f in image_files:
            images[f] = fits.getdata(f + '.fits')
            headers[f] = fits.getheader(f + '.fits')

    # Here we translate *all* catalog stars to x/y to display then and to 
    ifu_xy = project_xy(wdir, radec_file, fplane_file, tout['ra'], tout['dec'])
    # Read xy information, i.e. catalog derived x/y positions vs. actual detecion x/y
    t = ascii.read( os.path.join(wdir,xy_file) )
    matched = Table( [t['IFUSLOT_cat'], t['xifu_cat'], t['yifu_cat']], names=['ifuslot', 'xifu', 'yifu'])

    RMAX = 510.

    # Matrix
    ax_all = plt.axes([0.,0.,1/N,1/N])
    # next lines only to get a legend
    ax_all.plot([],[],'x',label="catalog",c='#2ECC71',markersize=10)
    ax_all.plot([],[],'r+',label="detected",markersize=10)
    ax_all.plot([],[],'o',label="matched",markersize=10, markerfacecolor='none', markeredgecolor='b')
    l = ax_all.legend()

    ax_all.xaxis.set_visible(False)
    ax_all.yaxis.set_visible(False)

    scaling = 1.8
    s = 51. * scaling

    fp = fplane.FPlane( os.path.join(wdir, fplane_file) )
    for f in images:
        ifuslot = f[-3:]

        if not ifuslot in fp.ifuslots:
            continue
        ifu = fp.by_ifuslot(ifuslot)
        x,y,xw,xy = (-(ifu.x) + RMAX - s/2)/N, (ifu.y - s/2 + RMAX)/N, s/N, s/N

        ax = plt.axes([x/RMAX/2.,y/RMAX/2.,xw/RMAX/2.,xy/RMAX/2.])

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        try:
            h = headers[f]
            xsize   = h['NAXIS1']
            ysize   = h['NAXIS2']
            xcenter = h['CRVAL1']
            ycenter = h['CRVAL2']
            extent = [0.+xcenter,xsize+xcenter,0.+ycenter,ysize+ycenter]

            ax.imshow( np.rot90(images[f], k=3), extent=extent, origin='bottom', vmin=-5., vmax=10., cmap=cmap)

            ii = ifu_xy['ifuslot'] == int(ifuslot)
            jj = matched['ifuslot'] == int(ifuslot)

            # don't get confused about rotations
            ax.plot(- ifu_xy['yifu'][ii], ifu_xy['xifu'][ii], 'x',c='#2ECC71',markersize=10)
            ax.plot(- matched['yifu'][jj], matched['xifu'][jj], 'o',markersize=10, markerfacecolor='none', markeredgecolor='b')

            ax.set_xlim([extent[0],extent[1]])
            ax.set_ylim([extent[2],extent[3]])
            dp = DAOPHOT_ALS.read( os.path.join(wdir, f + '.als') )
            ax.plot(- dp.data['Y']+51./2., dp.data['X']-51./2., 'r+',markersize=10)
            ax.text(.975,.025,ifuslot,transform=ax.transAxes,color='white',ha='right',va='bottom')
        except:
            pass


def get_exposures_files(basedir):
    """
    Create list of all file prefixes based
    on the existing collapsed IFU files in the current directory.

    From files:

    20180611T054545_015.fits
    ...
    20180611T054545_106.fits
    20180611T055249_015.fits
    ...
    20180611T055249_106.fits
    20180611T060006_015.fits
    ...
    20180611T060006_106.fits

    Creates:

    {
     'exp01' : ['20180611T054545_015',...,'20180611T054545_106']
     'exp02' : ['20180611T055249_015',...,'20180611T055249_106']
     'exp03' : ['20180611T060006_015',...,'20180611T060006_106']
    }


    Args:
       basedir (str): Directory to search.

    Returns:
       (OrderedDict): Ordered dictionary with pairs of exposure string "exp??" and time and list of
    """
    ff = []
    with path.Path(basedir):
        ff = glob.glob('2???????T??????_???.fits')
    _exp_datetimes = [f[:19] for f in ff]

    exp_datetimes = np.sort( np.unique([p[:15] for p in _exp_datetimes]) )

    exposures_files = OrderedDict()
    for i,edt in enumerate(exp_datetimes):
        files_for_exposure = []
        for f in ff:
            if f.startswith(edt):
                files_for_exposure.append(f.replace('.fits',''))
        exposures_files["exp{:02d}".format(i+1)] = files_for_exposure
    return exposures_files


def mk_match_plots(args, wdir, prefixes):
    """Creates match plots.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
        prefixes (list): List file name prefixes for the collapsed IFU images.
    """
    logging.info("mk_match_plots: Creating match plots.")

    shout_ifu_file = "shout.ifustars"
    exposures = ["exp01", "exp02", "exp03"]
    xy_files = {exp: "xy_{}.dat".format(exp) for exp in exposures}
    tmp_csv_files = {exp: "tmp_{}.csv".format(exp) for exp in exposures}
    radec_files = {exp: "radec2_{}.dat".format(exp) for exp in exposures}
    fplane_file = "fplane.txt"

    with path.Path(wdir):
        exposures_files = get_exposures_files(".")
        for exp in exposures:
            f = plt.figure(figsize=[15,15])
            ax = plt.subplot(111)
            if not exp in exposures_files:
                logging.warning("Found no image files for exposure {}.".format(exp))
                continue
            image_files = exposures_files[exp]
            xy_file = xy_files[exp]
            radec_file = radec_files[exp]
            if os.path.exists(xy_file) and os.path.exists(radec_file):
                mk_match_matrix(wdir, ax, exp, image_files, fplane_file, shout_ifu_file, xy_file, radec_file)
                f.savefig( "detect_{}.pdf".format(exp) )


def get_prefixes(wdir):
   """
   Create list of all file prefixes based
   on the existing collapsed IFU files in the current directory.


   Args:
       wdir (str): Work directory.
   """
   ff = []
   with path.Path(wdir):
       ff = glob.glob('2???????T??????_???.fits')
   return [f[:19] for f in ff]


def get_exposures(prefixes):
    """ Computes unique list of exposures from prefixes.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        prefixes (list): List file name prefixes for the collapsed IFU images

    Returns:
        (list): Unique list of exposure strings.
    """
    return np.unique([p[:15] for p in prefixes])


def cp_results(tmp_dir, results_dir):
    """ Coppies all relevant result files
    from tmp_dir results_dir.

    Args:
        tmp_dir (str): Temporary work directory.
        results_dir (str): Final directory for results.

    """
    dirs = ['add_radec_angoff_trial']
    file_pattern = []
#    file_pattern += ["CoFeS*_???_sci.fits"]
#    file_pattern += ["*.als"]
#    file_pattern += ["*.ap"]
#    file_pattern += ["*.coo"]
#    file_pattern += ["*.lst"]
#    file_pattern += ["2???????T??????_???.fits"]
    file_pattern += ["*.png"]
    file_pattern += ["all.mch"]
    file_pattern += ["all.raw"]
    file_pattern += ["allstar.opt"]
    file_pattern += ["daophot.opt"]
    file_pattern += ["fplane.txt"]
    file_pattern += ["getoff2_exp??.out"]
    file_pattern += ["getoff_exp??.out"]
    file_pattern += ["norm.dat"]
    file_pattern += ["photo.opt"]
    file_pattern += ["radec.orig"]
    file_pattern += ["radec2_exp??.dat"]
    file_pattern += ["radec_exp??.dat"]
#    file_pattern += ["shout.acamstars"]
    file_pattern += ["shout.ifu"]
    file_pattern += ["shout.ifustars"]
#    file_pattern += ["shout.info"]
#    file_pattern += ["shout.probestars"]
#    file_pattern += ["shout.result"]
    file_pattern += ["shuffle.cfg"]
#    file_pattern += ["tmp_exp??.csv"]
    file_pattern += ["use.psf"]
    file_pattern += ["2*fp.fits"]
    file_pattern += ["xy_exp??.dat"]
    file_pattern += ["detect_*.pdf"]
    file_pattern += ["radec2_final.dat"]
    file_pattern += ["radec2_final.pdf"]

    for d in dirs:
        td = os.path.join(tmp_dir,d)
        if os.path.exists(td):
            dir_util.copy_tree( td, results_dir)
    for p in file_pattern:
        ff = glob.glob("{}/{}".format(tmp_dir,p))
        for f in ff:
            shutil.copy2(f, results_dir)


def main():
    """
    Main function.
    """
    # Parse config file and command line paramters
    # command line parameters overwrite config file.
    args = parseArgs()

    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=args.logfile,
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


    # Create results directory for given night and shot
    cwd = os.getcwd()
    results_dir = os.path.join(cwd, "{}v{}".format(args.night, args.shotid))
    utils.createDir(results_dir)

    tasks = args.task.split(",")
    if args.use_tmp and not tasks == ['all']:
        logging.error("Step-by-step execution not possile when running tin a tmp directory.")
        logging.error("   Please either call without -t or set use_tmp to False.")
        sys.exit(1)

    # default is to work in results_dir
    wdir = results_dir
    if args.use_tmp:
        # Create a temporary directory
        tmp_dir = tempfile.mkdtemp()
        logging.info("Tempdir is {}".format(tmp_dir))
        logging.info("Copying over prior data (if any)...")
        dir_util.copy_tree(results_dir, tmp_dir)
        # set working directory to tmp_dir
        wdir = tmp_dir

    try:
        for task in tasks:
            if task in ["cp_post_stamps","all"]:
                # Copy over collapsed IFU cubes, aka IFU postage stamps.
                cp_post_stamps(args, wdir)

            prefixes  = get_prefixes(wdir)
            exposures = get_exposures(prefixes)

            if task in ["mk_post_stamp_matrix","all"]:
               # Creat IFU postage stamp matrix image.
               mk_post_stamp_matrix(args, wdir, prefixes)

            if task in ["daophot_find","all"]:
               # Run initial object detection in postage stamps.
               daophot_find(args, wdir, prefixes)

            if task in ["daophot_phot_and_allstar","all"]:
               # Run photometry 
               daophot_phot_and_allstar(args, wdir, prefixes)

            if task in ["mktot","all"]:
               # Combine detections accross all IFUs.
               mktot(args, wdir, prefixes)

            if task in ["rmaster","all"]:
              # Run daophot master to ???
              if len(exposures) > 1:
                rmaster(args, wdir)
              else:
                logging.info("Only one exposure, skipping rmaster.")

            if task in ["flux_norm","all"]:
               # Compute relative flux normalisation.
               if len(exposures) > 1:
                 flux_norm(args, wdir)
               else:
                 logging.info("Only one exposure, skipping flux_norm.")

            if task in ["redo_shuffle","all"]:
               # Rerun shuffle to get IFU stars
               redo_shuffle(args, wdir)

            if task in ["get_ra_dec_orig","all"]:
               # Retrieve original RA DEC from one of the multi files.
               # store in radec.orig
               get_ra_dec_orig(args, wdir)

            if task in ["compute_offset","all"]:
               # Compute offsets by matching 
               # detected stars to sdss stars from shuffle.
               # This also calls add_ra_dec to add RA DEC information to detections.
               compute_offset(args,wdir,prefixes)

            if task in ["compute_with_optimal_ang_off","all"]:
               # Compute offsets by matching 
               trial_dir =os.path.join(wdir, "add_radec_angoff_trial")
               optimal_ang_off= compute_optimal_ang_off(trial_dir,\
                       smoothing=args.optimal_ang_off_smoothing, PLOT=True)
               compute_offset(args, wdir,prefixes, final_ang_offset=optimal_ang_off)

            if task in ["combine_radec","all"]:
                # Combine individual exposure radec information.
                combine_radec(wdir)
            if task in ["add_ifu_xy","all"]:
               add_ifu_xy(args, wdir)

            if task in ["mkmosaic","all"]:
               # build mosaic for focal plane
               mkmosaic(args, wdir, prefixes)

            if task in ["mk_match_plots","all"]:
               # build mosaic for focal plane
               mk_match_plots(args, wdir, prefixes)

    finally:
        if args.use_tmp:
            logging.info("Copying over results.")
            cp_results(tmp_dir, results_dir)
            if args.remove_tmp:
                logging.info("Removing temporary directoy.")
                shutil.rmtree(tmp_dir)
        logging.info("Done.")


if __name__ == "__main__":
    sys.exit(main())
