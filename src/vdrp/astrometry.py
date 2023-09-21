#!/usr/bin/env python
""" Astrometry routine

Module to add astrometry to HETDEX catalgoues and images
Contains python translation of Karl Gebhardt

.. moduleauthor:: Maximilian Fabricius <mxhf@mpe.mpg.de>
"""

import matplotlib
matplotlib.use("agg")

from matplotlib import pyplot as plt

from numpy import loadtxt
from argparse import RawDescriptionHelpFormatter as ap_RDHF
from argparse import ArgumentParser as AP

import os
import glob
import shutil
import sys
import configparser
import logging
import subprocess
from astropy.io import fits
from astropy.io import ascii
import tempfile
import numpy as np
from collections import OrderedDict
import pickle
import ast
import re
import inspect

from scipy.interpolate import UnivariateSpline

from distutils import dir_util

import path
from astropy import table
from astropy.table import Table

from astropy.stats import biweight_location as biwgt_loc
from astropy.table import vstack
from astropy.io import ascii
from astropy.table import Table
from astropy.table import Column

from pyhetdex.het.fplane import FPlane
import pyhetdex.tools.read_catalogues as rc
from pyhetdex.het import fplane
from pyhetdex.coordinates.tangent_projection import TangentPlane
import pyhetdex.tools.read_catalogues as rc
# from pyhetdex import coordinates
from pyhetdex.coordinates import astrometry as phastrom

from vdrp.cofes_vis import cofes_4x4_plots
from vdrp import daophot
from vdrp import cltools
from vdrp import utils
from vdrp.daophot import DAOPHOT_ALS
from vdrp.utils import read_radec, write_radec
from vdrp.fplane_client import retrieve_fplane
from vdrp.vdrp_helpers import VdrpInfo


def getDefaults():

    defaults = {}
    defaults["use_tmp"] = "False"
    defaults["remove_tmp"] = "True"
    defaults["logfile"] = "astrometry.log"
    defaults['tmp_dir'] = '/tmp/'
    defaults["reduction_dir"] = "/work/03946/hetdex/maverick/red1/reductions/"
    defaults["cofes_vis_vmin"] = -15.
    defaults["cofes_vis_vmax"] = 25.
    defaults["daophot_sigma"] = 2
    defaults["daophot_xmin"] = 4
    defaults["daophot_xmax"] = 45
    defaults["daophot_ymin"] = 4
    defaults["daophot_ymix"] = 45
    defaults["daophot_opt"] = "$config/daophot.opt"
    defaults["daophot_phot_psf"] = "$config/use.psf"
    defaults["daophot_photo_opt"] = "$config/photo.opt"
    defaults["daophot_allstar_opt"] = "$config/allstar.opt"
    defaults["mktot_ifu_grid"] = "$config/ifu_grid.txt"
    defaults["mktot_magmin"] = 0.
    defaults["mktot_magmax"] = 21.
    defaults["mktot_xmin"] = 0.
    defaults["mktot_xmax"] = 50.
    defaults["mktot_ymin"] = 0.
    defaults["mktot_ymax"] = 50.
    defaults["fluxnorm_mag_max"] = 19.
    defaults["fplane_txt"] = "$config/fplane.txt"
    defaults["shuffle_cfg"] = "$config/shuffle.cfg"
    defaults["acam_magadd"] = 5.
    defaults["wfs1_magadd"] = 5.
    defaults["wfs2_magadd"] = 5.
    defaults["add_radec_angoff"] = 0.1
    defaults["add_radec_angoff_trial"] = \
        "1.35,1.375,1.4,1.425,1.45,1.475,1.5,1.525,1.55,1.575,1.6"
    defaults["add_radec_angoff_trial_dir"] = "add_radec_angoff_trial"
    defaults["getoff2_radii"] = '11., 5., 3.'
    defaults["mkmosaic_angoff"] = 1.8
    defaults["task"] = "all"
    defaults["offset_exposure_indices"] = "1,2,3"
    defaults["optimal_ang_off_smoothing"] = 0.05
    defaults["dither_offsets"] = "[(0.,0.),(1.270,-0.730),(1.270,0.730)]"
    # for fibcoords
    defaults["parangle"] = -999999.

    return defaults


def parseArgs(args):
    """ Parses configuration file and command line arguments.
    Command line arguments overwrite configuration file settiongs which
    in turn overwrite default values.

    Parameters
    ----------
    args : argparse.Namespace
        Return the populated namespace.
    """

    # Parse any conf_file specification
    # We make this parser with add_help=False so that
    # it doesn't parse -h and print help.
    conf_parser = AP(description=__doc__,  # printed with -h/--help
                     # Don't mess with format of description
                     formatter_class=ap_RDHF,
                     # Turn off help, so we print all options in response to -h
                     add_help=False)
    conf_parser.add_argument("-c", "--conf_file",
                             help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()

    defaults = getDefaults()

    config_source = "Default"
    if args.conf_file:
        config_source = args.conf_file
        config = configparser.ConfigParser()
        config.read([args.conf_file])
        defaults.update(dict(config.items("Astrometry")))

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h

    # Inherit options from config_paarser
    parser = AP(parents=[conf_parser])

    parser.set_defaults(**defaults)
    parser.add_argument("--logfile", type=str, help="Filename for log file.")
    parser.add_argument("--use_tmp", type=str,
                        help="Use a temporary directory. Result files will "
                        "be copied to NIGHTvSHOT.")
    parser.add_argument("--remove_tmp", type=str,
                        help="Remove temporary directory after completion.")
    parser.add_argument("--tmp_dir", type=str, help="Base directory "
                        "used to create the temporary work directory")
    parser.add_argument("--reduction_dir", type=str,
                        help="Directory that holds panacea reductions. "
                        "Subdriectories with name like NIGHTvSHOT must exist.")
    parser.add_argument("--cofes_vis_vmin", type=float,
                        help="Minimum value (= white) "
                        "for matrix overview plot.")
    parser.add_argument("--cofes_vis_vmax", type=float,
                        help="Maximum value (= black) "
                        "for matrix overview plot.")
    parser.add_argument("--daophot_sigma", type=float,
                        help="Daphot sigma value.")
    parser.add_argument("--daophot_xmin", type=float,
                        help="X limit for daophot detections.")
    parser.add_argument("--daophot_xmax", type=float,
                        help="X limit for daophot detections.")
    parser.add_argument("--daophot_ymin", type=float,
                        help="Y limit for daophot detections.")
    parser.add_argument("--daophot_ymix", type=float,
                        help="Y limit for daophot detections.")
    parser.add_argument("--daophot_phot_psf", type=str,
                        help="Filename for daophot PSF model.")
    parser.add_argument("--daophot_opt", type=str,
                        help="Filename for daophot configuration.")
    parser.add_argument("--daophot_photo_opt", type=str,
                        help="Filename for daophot photo task configuration.")
    parser.add_argument("--daophot_allstar_opt", type=str,
                        help="Filename for daophot "
                        "allstar task configuration.")
    parser.add_argument("--mktot_ifu_grid", type=str,
                        help="Name of file that holds grid of "
                        "IFUs offset fit (mktot).")
    parser.add_argument("--mktot_magmin", type=float,
                        help="Magnitude limit for offset fit (mktot).")
    parser.add_argument("--mktot_magmax", type=float,
                        help="Magnitude limit for offset fit (mktot).")
    parser.add_argument("--mktot_xmin", type=float,
                        help="X limit for offset fit (mktot).")
    parser.add_argument("--mktot_xmax", type=float,
                        help="X limit for offset fit (mktot).")
    parser.add_argument("--mktot_ymin", type=float,
                        help="Y limit for offset fit (mktot).")
    parser.add_argument("--mktot_ymax", type=float,
                        help="Y limit for offset fit (mktot).")
    parser.add_argument("--fluxnorm_mag_max", type=float,
                        help="Magnitude limit for flux normalisation.")
    parser.add_argument("--fplane_txt", type=str,
                        help="filename for fplane file.")
    parser.add_argument("--shuffle_cfg", type=str,
                        help="Filename for shuffle configuration.")
    parser.add_argument("--acam_magadd", type=float,
                        help="do_shuffle acam magadd.")
    parser.add_argument("--wfs1_magadd", type=float,
                        help="do_shuffle wfs1 magadd.")
    parser.add_argument("--wfs2_magadd", type=float,
                        help="do_shuffle wfs2 magadd.")
    parser.add_argument("--add_radec_angoff", type=float,
                        help="Angular offset to add during conversion of x/y "
                        "coordinate to RA/Dec.")
    parser.add_argument("--add_radec_angoff_trial", type=str,
                        help="Trial values for angular offsets.")
    parser.add_argument("--add_radec_angoff_trial_dir", type=str,
                        help="Directory to save results of angular offset "
                        "trials.")
    parser.add_argument('--getoff2_radii', type=str,
                        help="Comma separated list of matching radii for "
                        "astrometric offset measurement.")
    parser.add_argument("--mkmosaic_angoff", type=float,
                        help="Angular offset to add for creation of "
                        "mosaic image.")
    parser.add_argument("-t", "--task", type=str, help="Task to execute.")
    parser.add_argument("--offset_exposure_indices", type=str,
                        help="Exposure indices.")
    parser.add_argument("--optimal_ang_off_smoothing", type=float,
                        help="Smothing value for smoothing spline use for"
                        " measurement of optimal angular offset value.")
    parser.add_argument("--dither_offsets", type=str,
                        help="List of x,y tuples that define the "
                        "dither offsets.")
    parser.add_argument("--parangle", type=float,
                        help="Optional parangle to use if the one found"
                        "in the header is unknown (-999999.).")
    parser.add_argument("--shifts_dir", type=str)

    # positional arguments
    parser.add_argument('night', metavar='night', type=str,
                        help='Night of observation (e.g. 20180611).')
    parser.add_argument('shotid', metavar='shotid', type=str,
                        help='Shot ID (e.g. 017).')
    parser.add_argument('ra', metavar='ra', type=float,
                        help='RA of the target in decimal hours.', nargs='?',
                        default=None)
    parser.add_argument('dec', metavar='dec', type=float,
                        help='Dec of the target in decimal hours degree.',
                        nargs='?', default=None)
    parser.add_argument('track', metavar='track', type=int, choices=[0, 1],
                        help='Type of track: 0: East 1: West', nargs='?',
                        default=None)

    args = parser.parse_args(remaining_argv)

    args.config_source = config_source
    # should in principle be able to do this with accumulate???
    args.use_tmp = args.use_tmp == "True"
    args.remove_tmp = args.remove_tmp == "True"
    args.getoff2_radii = [float(t) for t in args.getoff2_radii.split(",")]
    args.add_radec_angoff_trial = [float(offset) for offset in
                                   args.add_radec_angoff_trial.split(",")]
    args.dither_offsets = ast.literal_eval(args.dither_offsets)

    args.offset_exposure_indices = [int(t) for t in
                                    args.offset_exposure_indices.split(",")]

    args.daophot_opt = utils.mangle_config_pathname(args.daophot_opt)
    args.daophot_phot_psf = \
        utils.mangle_config_pathname(args.daophot_phot_psf)
    args.daophot_photo_opt = \
        utils.mangle_config_pathname(args.daophot_photo_opt)
    args.daophot_allstar_opt = \
        utils.mangle_config_pathname(args.daophot_allstar_opt)
    args.mktot_ifu_grid = utils.mangle_config_pathname(args.mktot_ifu_grid)
    args.fplane_txt = utils.mangle_config_pathname(args.fplane_txt)
    args.shuffle_cfg = utils.mangle_config_pathname(args.shuffle_cfg)

    return args


def cp_post_stamps(wdir, reduction_dir, night, shotid):
    """ Copy CoFeS (collapsed IFU images).

    Parameters
    ----------
    wdir : str
        Work directory.
    reduction_dir : str
        Directory that holds panacea reductions.
    night : str
        Night (e.g. 20180611)
    shotid : str
        ID of shot (e.g. 017)

    Raises
    ------
        Exception
    """
    # find the IFU postage stamp fits files and copy them over
    pattern = os.path.join(reduction_dir,
                           "{}/virus/virus0000{}/*/*/Co*".format(night,
                                                                 shotid))
    logging.info("Copy {} files to {}".format(pattern, wdir))
    cofes_files = glob.glob(pattern)
    if len(cofes_files) == 0:
        raise Exception("Found no postage stamp images. Please check your "
                        "reduction_dir in config file.")
    already_warned = False
    for f in cofes_files:
        h, t = os.path.split(f)
        target_filename = t[5:20] + t[22:26] + ".fits"
        if os.path.exists(os.path.join(wdir, target_filename)):
            if not already_warned:
                logging.warning("{} already exists in {}, skipping, won't warn"
                                " about other "
                                "files....".format(target_filename,
                                                   wdir))
                already_warned = True
            continue

        shutil.copy2(f, os.path.join(wdir, target_filename))


def mk_post_stamp_matrix(wdir, prefixes, cofes_vis_vmin, cofes_vis_vmax):
    """ Create the IFU postage stamp matrix image.

    Parameters
    ----------
    wdir : str
        Work directory.
    prefixes : list
        List file name prefixes for the collapsed IFU images.
    cofes_vis_vmin : float
        Minimum value (= black) for matrix overview plot.
    cofes_vis_vmax : float
        Maximum value (= black) for matrix overview plot.
    """
    # create the IFU postage stamp matrix image
    logging.info("Creating the IFU postage stamp matrix images ...")
    exposures = np.unique([p[:15] for p in prefixes])

    with path.Path(wdir):
        for exp in exposures:
            outfile_name = exp + ".png"
            logging.info("Creating {} ...".format(outfile_name))
            cofes_4x4_plots(prefix=exp, outfile_name=outfile_name,
                            vmin=cofes_vis_vmin, vmax=cofes_vis_vmax,
                            logging=logging)


def daophot_find(wdir, prefixes, daophot_opt, daophot_sigma, daophot_xmin,
                 daophot_xmax, daophot_ymin, daophot_ymix):
    """ Run initial daophot find.

    Parameters
    ----------
    wdir : str
        Work directory.
    prefixes : list
        List file name prefixes for the collapsed IFU images.
    daophot_opt : str
        Daphot sigma value.
    daophot_sigma : float
        Filename for daophot configuration.
    daophot_xmin : float
        X limit for daophot detections.
    daophot_xmax : float
        X limit for daophot detections.
    daophot_ymin : float
        Y limit for daophot detections.
    daophot_ymix : float
        Y limit for daophot detections.
    """
    logging.info("Running initial daophot find...")
    # Create configuration file for daophot.
    shutil.copy2(daophot_opt, os.path.join(wdir, "daophot.opt"))
    with path.Path(wdir):
        for prefix in prefixes:
            # execute daophot
            daophot.daophot_find(prefix, daophot_sigma, logging=logging)
            # filter ouput
            daophot.filter_daophot_out(prefix+".coo", prefix+".lst",
                                       daophot_xmin, daophot_xmax,
                                       daophot_ymin, daophot_ymix)


def daophot_phot_and_allstar(wdir, prefixes, daophot_photo_opt,
                             daophot_allstar_opt, daophot_phot_psf):
    """ Runs daophot photo and allstar on all IFU postage stamps.
    Produces \*.ap and \*.als files.
    Analogous to run4a.

    Parameters
    ----------
    wdir : str
        Work directory.
    prefixes : list
        List file name prefixes for the collapsed IFU images.
    daophot_opt : str
        Filename for daophot configuration.
    daophot_photo_opt : str
        Filename for daophot photo task configuration.
    daophot_allstar_opt : str
        Filename for daophot allstar task configuration.

    """
    # run initial daophot phot & allstar
    logging.info("Running daophot phot & allstar ...")
    # Copy configuration files for daophot and allstar.
    shutil.copy2(daophot_photo_opt, os.path.join(wdir, "photo.opt"))
    shutil.copy2(daophot_allstar_opt, os.path.join(wdir, "allstar.opt"))
    shutil.copy2(daophot_phot_psf, os.path.join(wdir, "use.psf"))
    with path.Path(wdir):
        for prefix in prefixes:
            # first need to shorten file names such
            # that daophot won't choke on them.
            daophot.daophot_phot(prefix, logging=logging)
            daophot.allstar(prefix, logging=logging)


def mktot(wdir, prefixes, mktot_ifu_grid, mktot_magmin, mktot_magmax,
          mktot_xmin, mktot_xmax, mktot_ymin, mktot_ymax, dither_offsets):
    """ Reads all *.als files. Put detections on a grid
    corresponding to the IFU position in the focal plane as defined in
    config/ifu_grid.txt (should later become fplane.txt.
    Then produces all.mch.

    Notes
    -----
        Analogous to run6 and run6b.

    Parameters
    ----------
    wdir : str
        Work directory.
    prefixes : list
        List file name prefixes for the collapsed IFU images.
    mktot_ifu_grid : str
        Name of file that holds gird of IFUs offset fit (mktot).
    mktot_magmin : float
        Magnitude limit for offset fit.
    mktot_magmax : float
        Magnitude limit for offset fit.
    mktot_xmin : float
        X limit for offset fit.
    mktot_xmax : float
        X limit for offset fit.
    mktot_ymin : float
        Y limit for offset fit.
    mktot_ymax : float
        Y limit for offset fit.

    """
    # read IFU grid definition file (needs to be replaced by fplane.txt)
    ifugird = Table.read(mktot_ifu_grid, format='ascii')

    with path.Path(wdir):
        exposures = np.unique([p[:15] for p in prefixes])

        for exp in exposures:
            fnout = exp + "tot.als"
            logging.info("Creating {}".format(fnout))

            with open(fnout, 'w') as fout:
                s = " NL   NX   NY  LOWBAD HIGHBAD  THRESH     AP1  PH/ADU  RNOISE    FRAD\n"
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
                        logging.warning("IFU slot {} not found in "
                                        "{}.".format(ifuslot, mktot_ifu_grid))
                        continue
                    ifugird['X'][jj][0]
                    ifugird['Y'][jj][0]
                    xoff = ifugird['X'][jj][0]
                    yoff = ifugird['Y'][jj][0]

                    # read daophot als input file
                    try:
                        als = daophot.DAOPHOT_ALS.read(prefix + ".als")
                    except Exception:
                        logging.warning("Unable to read " + prefix + ".als")
                        continue

                    # filter according to magnitude and x and y range
                    ii = als.data['MAG'] > mktot_magmin
                    ii *= als.data['MAG'] < mktot_magmax
                    ii *= als.data['X'] > mktot_xmin
                    ii *= als.data['X'] < mktot_xmax
                    ii *= als.data['Y'] > mktot_ymin
                    ii *= als.data['Y'] < mktot_ymax

                    count += sum(ii)

                    for d in als.data[ii]:
                        # s = "{:03d} {:8.3f} {:8.3f} " \
                        #     "{:8.3f}\n".format(d['ID'], d['X']+xoff,
                        #                        d['Y']+yoff, d['MAG'])
                        s = "{:d} {:8.3f} {:8.3f} " \
                            "{:8.3f}\n".format(d['ID'], d['X']+xoff,
                                               d['Y']+yoff, d['MAG'])
                        fout.write(s)

                logging.info("{} stars in {}.".format(count, fnout))
        # produce all.mch like run6b
        with open("all.mch", 'w') as fout:
            s = ""
            for i in range(len(exposures)):
                s += " '{:30s}'     {:.3f}     {:.3f}   1.00000   0.00000  " \
                    " 0.00000   1.00000     0.000    " \
                    "0.0000\n".format(exposures[i] + "tot.als",
                                      dither_offsets[i][0],
                                      dither_offsets[i][1])
            fout.write(s)


def rmaster(wdir):
    """ Executes daomaster. This registers the sets of detections
    for the thre different exposrues with respec to each other.

    Notes
    -----
        Analogous to run8b.

    Parameters
    ----------
    wdir : str
        Work directory.
    """
    logging.info("Running daomaster.")

    with path.Path(wdir):
        daophot.rm(["all.raw"])
        daophot.daomaster(logging=logging)


def getNorm(all_raw, mag_max):
    """ Comutes the actual normalisation for flux_norm.

    Notes
    -----
        Analogous to run9.

    Parameters
    ----------
    all_raw : str
        Output file name of daomaster, usuall all.raw.
    mag_max : float
        Magnitude cutoff for normalisation.
        Fainter objects will be ignored.
    """
    def mag2flux(m):
        return 10**((25-m)/2.5)

    ii = all_raw[:, 3] < mag_max
    ii *= all_raw[:, 5] < mag_max
    ii *= all_raw[:, 7] < mag_max

    f1 = mag2flux(all_raw[ii, 3])
    f2 = mag2flux(all_raw[ii, 5])
    f3 = mag2flux(all_raw[ii, 7])

    favg = (f1+f2+f3)/3.
    return biwgt_loc(f1/favg), biwgt_loc(f2/favg), biwgt_loc(f3/favg)


def flux_norm(wdir, mag_max, infile='all.raw', outfile='norm.dat'):
    """ Reads all.raw and compute relative flux normalisation
    for the three exposures.

    Notes
    -----
        Analogous to run9.

    Parameters
    ----------
    wdir : str
        Work directory.
    mag_max : float
        Magnitude limit for flux normalisation.
    infile : str
        Output file of daomaster.
    outfile : str
        Filename for result file.
    """
    global vdrp_info
    logging.info("Computing flux normalisation between exposures 1,2 and 3.")
    with path.Path(wdir):
        all_raw = loadtxt(infile, skiprows=3)
        n1, n2, n3 = getNorm(all_raw, mag_max)
        vdrp_info["fluxnorm_exp1"] = n1
        vdrp_info["fluxnorm_exp2"] = n2
        vdrp_info["fluxnorm_exp3"] = n3
        logging.info("flux_norm: Flux normalisation is {:10.8f} {:10.8f} "
                     "{:10.8f}".format(n1, n2, n3))
        with open(outfile, 'w') as f:
            s = "{:10.8f} {:10.8f} {:10.8f}\n".format(n1, n2, n3)
            f.write(s)


def redo_shuffle(wdir, ra, dec, track, acam_magadd, wfs1_magadd, wfs2_magadd,
                 shuffle_cfg, fplane_txt, night, catalog=None):
    """
    Reruns shuffle to obtain catalog of IFU stars.

    Creates a number of output files, most importantly
    `shout.ifustars` which is used as catalog for the offset computation.

    Parameters
    ----------
    wdir : str
        Work directory.
    ra : float
        Right ascension in degrees.
    dec : float
        Declination in degrees.
    track : int
        East or west track (0, 1)
    acam_magadd : float
        do_shuffle acam magadd.
    wfs1_magadd : float
        do_shuffle wfs1 magadd.
    wfs2_magadd : float
        do_shuffle wfs2 magadd.
    """
    logging.info("Using {}.".format(shuffle_cfg))
    shutil.copy2(shuffle_cfg, os.path.join(wdir, "shuffle.cfg"))
    retrieve_fplane(night, fplane_txt, wdir)
    with path.Path(wdir):
        try:
            os.remove('shout.ifustars')
        except Exception:
            pass

        RA0 = ra
        DEC0 = dec
        radius = 0.
        track = track
        ifuslot = 0
        x_offset = 0.
        y_offset = 0

        daophot.rm(['shout.acamstars', 'shout.ifustars', 'shout.info',
                    'shout.probestars', 'shout.result'])
        logging.info("Rerunning shuffle for RA = {}, Dec = {} and "
                     "track = {} ...".format(RA0, DEC0, track))
        cmd = "do_shuffle -v --acam_magadd {:.2f} --wfs1_magadd {:.2f}" \
            " --wfs2_magadd {:.2f} ".format(acam_magadd, wfs1_magadd,
                                            wfs2_magadd)
        if catalog is not None:
            cmd += "--catalog {} ".format(catalog)
        cmd += " {:.6f} {:.6f} {:.1f} {:d} {:d} {:.1f} " \
            "{:.1f}".format(RA0, DEC0, radius, track, ifuslot,
                            x_offset, y_offset)
        logging.info("Calling shuffle with {}".format(cmd))
        subprocess.call(cmd, shell=True)

        # archive the result
        shutil.copy2('shout.ifustars', 'sdss.{}'.format(catalog))


def get_track(wdir, reduction_dir, night, shotid):
    """
    Reads first of the many multi* file'd headers to get
    the track.

    Notes
    -----
        This function is so emparrisingly similar to get_ra_dec_orig
        that they should probably be combined.

    Parameters
    ----------
    wdir : str
        Work directory.
    reduction_dir : str
        Directory that holds panacea reductions.
    night : str
        Night (e.g. 20180611)
    shotid : str
        ID of shot (e.g. 017)

    Returns
    ------
        (int): 0 = east track, 1 = west track
    """
    global vdrp_info
    pattern = \
        os.path.join(reduction_dir,
                     "{}/virus/virus0000{}/*/*/multi_???_*LL*fits"
                     .format(night, shotid))
    multi_files = glob.glob(pattern)
    if len(multi_files) == 0:
        raise Exception("Found no multi file in {}. Please check "
                        "reduction_dir in configuration file."
                        .format(reduction_dir))
    h = fits.getheader(multi_files[0])
    az = h["STRUCTAZ"]
    logging.info("STRUCTAZ = {}".format(az))
    track = None
    if az < 180.:
        track = 0
    else:
        track = 1

    logging.info("-> track = {}".format(track))

    if vdrp_info is not None:
        vdrp_info["STRUCTAZ"] = az
        vdrp_info["track"] = track

    return track


def get_ra_dec_orig(wdir, reduction_dir, night, shotid, user_pa=-999999.):
    """
    Reads first of the many multi* file'd headers to get
    the RA, DEC, PA guess from the telescope.

    Notes
    -----
        Creates radec.orig

    Parameters
    ----------
    wdir : str
        Work directory.
    reduction_dir : str
        Directory that holds panacea reductions.
    night : str
        Night (e.g. 20180611)
    shotid : str
        ID of shot (e.g. 017)
    """
    global vdrp_info
    pattern = \
        os.path.join(reduction_dir,
                     "{}/virus/virus0000{}/*/*/multi_???_*LL*fits"
                     .format(night, shotid))
    multi_files = glob.glob(pattern)
    if len(multi_files) == 0:
        raise Exception("Found no multi file in {}. Please check "
                        "reduction_dir in configuration file."
                        .format(reduction_dir))
    h = fits.getheader(multi_files[0])
    ra0 = h["TRAJRA"]
    dec0 = h["TRAJDEC"]
    pa0 = h["PARANGLE"]
    if pa0 < -999990.:  # Unknown value in header, use optional user value
        pa0 = user_pa
    logging.info("Original RA,DEC,PA = {},{},{}".format(ra0, dec0, pa0))

    if vdrp_info is not None:
        vdrp_info["orig_ra"] = ra0
        vdrp_info["orig_dec"] = dec0
        vdrp_info["orig_pa0"] = pa0

    utils.write_radec(ra0, dec0, pa0, os.path.join(wdir, "radec.orig"))


def get_als_files(fp, exp_prefixes):
    """
    Derives for a list of exposure prefixes a list
    of \*.als files, but rejects any that refer to an IFU slot
    which is not contained in the fplane.

    Parameters
    ----------
    fp : pyhetdex.het.fplane.FPlane
        Fplane object.
    exp_prefixes : list
        List of epxosure prefixes.

    Returns
    -------
        (list): List of *.als files.
    """
    # collect als files for all IFUs that are contained in the fplane file.
    als_files = []
    for prefix in exp_prefixes:
        ifuslot = prefix[-3:]
        if ifuslot not in fp.ifuslots:
            logging.warning("IFU slot {} not contained in fplane.txt."
                            .format(ifuslot))
            continue
        fn = prefix + ".als"
        als_files.append(fn)
    return als_files


def load_als_data(als_files):
    """ Load set of als files.

    Parameters
    ----------
    als_files : list
        List of file names.

    Returns
    ------
        (OrderedDict):  Dictionary with als data for each IFU slot.
    """
    # work out the IFU slot from the file name
    als_data = OrderedDict()
    for fn in als_files:
        ihmp = fn[-7:-4]
        data = rc.read_daophot(fn)
        als_data[ihmp] = data
    return als_data


def add_ra_dec(wdir, als_data, ra, dec, pa, fp, radec_outfile='tmp.csv'):
    """
    Call add_ra_dec to compute for detections in IFU space the
    corresponding RA/DEC coordinates.

    New version, direct python call to pyheted.coordinates.tangent_projection.

    Requires, fplane.txt
    Creates primarely EXPOSURE_tmp.csv but also radec.dat.

    Parameters
    ----------
    wdir : str
        Work directory.
    als_data : dict
        Dictionary with als data for each IFU slot.
    ra : float
        Focal plane center RA.
    dec : float
        Focal plane center Dec.
    pa : float
        Positions angle.
    fp : FPlane
        Focal plane object.
    radec_outfile : str
        Filename that will contain output from
        add_ra_dec (gets overwritten!).

    """
    with path.Path(wdir):
        fp = fplane.FPlane("fplane.txt")

        # Carry out required changes to astrometry
        rot = 360.0 - (pa + 90.)
        # Set up astrometry from user supplied options
        tp = TangentPlane(ra, dec, rot)

        # Loop over the files
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
        table_out.write(radec_outfile, comment='#', overwrite=True)


def compute_optimal_ang_off(wdir, smoothing=0.05, PLOT=True):
    """
    Computes the optimal angular offset angle by findin the minimal
    RMS of a set of different trial angles.

    Takes (if exist) all three different exposures into account and computes
    weighted average ange (weighted by number of stars that went into the fit).

    The RMS(ang_off) are interpolate with a smoothing spline.
    The smoothing value is a parameter to this function.

    Parameters
    ----------
    wdir : str
        Directory that holds the angular offset trials
        (e.g. 20180611v017/add_radec_angoff_trial)

    Returns
    -------
        float : Optimal offset angle.
    """
    global vdrp_info
    colors = ['red', 'green', 'blue']
    exposures = ['exp01', 'exp02', 'exp03']

    logging.info("Computing optimal angular offset...")
    # load getoff2 data for all exposures
    results = Table(names=['exposure', 'ang_off', 'nstar', 'RMS'],
                    dtype=['S5', float, int, float])
    for exp in exposures:
        list = glob.glob("{}/getoff2_{}*Deg.out".format(wdir, exp))
        if len(list) == 0:
            logging.warning("Found no files for exposure {}".format(exp))
            continue

        for filename in list:
            # count how many stars contribute
            with open(filename.replace('getoff2', 'getoff')) as f:
                ll = f.readlines()
            nstar = len(ll)
            # now load the getoff2 to read the RMS
            ang = float(filename.replace('Deg.out', '').split("_")[-1])
            with open(filename) as f:
                ll = f.readlines()
            try:
                tt = ll[0].split()
                rms_dra = float(tt[2])
                rms_ddec = float(tt[3])
                results.add_row([exp, ang, nstar,
                                 np.sqrt(rms_dra**2. + rms_ddec**2.)])
            except Exception:
                logging.error("Parsing {}".format(filename))

    if len(results) == 0:
        logging.error("Found no data for angular offset trials.")
        return np.nan

    if PLOT:
        fig = plt.figure(figsize=[7, 7])
        ax = plt.subplot(111)

    # angular subgrid for interpolation
    aa = np.arange(results['ang_off'].min(), results['ang_off'].max(), 0.01)
    aamin = Table(names=["exposure", "ang_off_min", "nstar_min", "RMS_min"],
                  dtype=['S5', float, int, float])
    # iterate over all 1-3 exposures.
    for i, exp in enumerate(exposures):
        ii = results['exposure'] == exp
        if sum(ii) <= 3:
            logging.warning("Insufficient data for exposure {}.".format(exp))
            continue

        x = results['ang_off'][ii]
        y = results['RMS'][ii]
        n = results['nstar'][ii]
        jj = np.argsort(x)

        x = x[jj]
        y = y[jj]
        n = n[jj]

        # old cubic interpolation
        # f = interpolate.interp1d(x, y, kind='cubic', bounds_error=False)
        f = UnivariateSpline(x, y, s=smoothing)
        # this is a bit silly, but since the number of stars may
        # change as a function of
        # angle we also need to interpolate those.
        fn = UnivariateSpline(x, n, s=smoothing)
        # find best offset angle
        imin = np.nanargmin(f(aa))
        amin = aa[imin]
        rms_min = f(aa[imin])
        nstar_min = fn(aa[imin])
        aamin.add_row([exp, amin, nstar_min, rms_min])
        vdrp_info["ang_off_{}".format(exp)] = amin

        if PLOT:
            plt.plot(x, y, 'o', c=colors[i], label=exp)
            plt.plot(aa, f(aa), '-', c=colors[i])
            plt.axvline(amin, color=colors[i])
            plt.text(amin, 1.5, "{:.3f} Deg # stars = {}"
                     .format(amin, nstar_min), color=colors[i], rotation=90.,
                     ha='right')

    # average optimal offset angle accross all exposures
    ang_off_avg = np.sum(aamin['ang_off_min']*aamin['nstar_min']) \
        / np.sum(aamin['nstar_min'])

    if PLOT:
        plt.axvline(ang_off_avg, color="k")
        plt.legend(loc="lower right")
        plt.xlabel("Offset angle")
        plt.ylabel("RMS")
        plt.text(.1, .9, "avg. optimal\noffset angle\nis {:.5} Deg"
                 .format(ang_off_avg), transform=ax.transAxes)
        fig.tight_layout()
        plt.savefig(os.path.join(wdir, "ang_off.pdf"), overwrite=True)

    vdrp_info["ang_off_avg"] = ang_off_avg
    return ang_off_avg


def compute_offset(wdir, prefixes, getoff2_radii, add_radec_angoff_trial,
                   add_radec_angoff, add_radec_angoff_trial_dir,
                   offset_exposure_indices, final_ang_offset=None,
                   shout_ifustars='shout.ifustars', ra0=None, dec0=None):
    """
    Requires, fplane.txt and radec.orig. If not ra, dec are passed
    explicitly then the values from radec.orig are used. The
    pa value from radec.orig is used in any case.
    Creates primarely EXPOSURE_tmp.csv but also radec2.dat.

    Compute offset in RA DEC  by matching detected stars in IFUs
    against the shuffle profived RA DEC coordinates.

    Notes
    -----
        Analogous to rastrom3.
        Creates radec.dat, radec2.dat and
        radec_TRIAL_OFFSET_ANGLE.dat, radec_TRIAL_OFFSET_ANGLE2.dat.

    Parameters
    ----------
    wdir : str
        Work directory.
    prefixes : list
        List file name prefixes for the collapsed IFU images.
    getoff2_radii : list
        List of matching radii for astrometric offset measurement.
    add_radec_angoff_trial : list
        Trial values for angular offsets.
    add_radec_angoff : float
        Angular offset to add during conversion
        of x/y coordinate to RA/Dec.
    add_radec_angoff_trial_dir : str
        Directory to save results of angular offset trials.
    offset_exposure_indices : list
        Exposure indices.
    final_ang_offset : float
        Final angular offset to use. This overwrites the values in
        add_radec_angoff and add_radec_angoff_trial
    shout_ifustars : str
        Shuffle output catalog of IFU stars.
    ra0 : float
        Optionally allows to overwrite use of RA from radec.orig
    dec0 : float
        Optionally allows to overwrite use of DEC from radec.orig
    """
    global vdrp_info

    def write_ra_dec_dats(ra, dec, pa, exp_index, angoff,
                          ra_offset, dec_offset, nominal=False):
        sangoff = ""
        if not nominal:
            sangoff = '_{:06.3f}Deg'.format(angoff)

        # write results to radec_exp??.dat
        fnout = "radec_exp{:02d}{}.dat".format(exp_index, sangoff)
        utils.write_radec(ra*15., dec, pa + angoff, fnout)
        logging.info("Wrote {}".format(fnout))

        # write results to radec2_exp??.dat
        fnout = "radec2_exp{:02d}{}.dat".format(exp_index, sangoff)
        utils.write_radec(ra*15. + ra_offset, dec + dec_offset, pa + angoff,
                          fnout)
        logging.info("Wrote {}".format(fnout))

    with path.Path(wdir):
        radii = getoff2_radii

        # Here we iterate over all angular offset angles
        # as configured in the config file, parameter add_radec_angoff_trial.
        # Should add_radec_angoff not be in that list, we add it here
        # and the third line makes sure that the nominal angle
        # is the last one that we compute. This is important
        # such that all the correct output files are in place
        # for the downstream functions.
        angoffsets = add_radec_angoff_trial
        nominal_angoffset = add_radec_angoff
        if final_ang_offset is not None:
            logging.info("Using final angular offset value of {} Deg."
                         .format(final_ang_offset))
            angoffsets = []
            nominal_angoffset = final_ang_offset
            add_radec_angoff = final_ang_offset

        angoffsets = [x for x in angoffsets if x != nominal_angoffset] \
            + [nominal_angoffset]

        # Give comprehensive information about the iterations.
        s = ""
        for r in radii:
            s += "{}\" ".format(r)
        logging.info("Computing offsets with using following sequence "
                     "of matching radii: {}".format(s))
        s = ""
        for a in add_radec_angoff_trial:
            s += "{} Deg ".format(a)
        if final_ang_offset is None:
            logging.info("Also computing offsets for the following set "
                         "of trial angles: {}".format(s))

        # will contain results of angular offset trials
        utils.createDir(add_radec_angoff_trial_dir)

        for angoff in angoffsets:
            # collect the prefixes that belong to the first exposure
            # for now only do first exposure, later can do them all
            exposures = np.sort(np.unique([p[:15] for p in prefixes]))

            # loop over all exposures in configuration file
            for exp_index in offset_exposure_indices:
                if exp_index > len(exposures):
                    logging.warning("Have no data for exposure {}. "
                                    "Skipping ...".format(exp_index))
                    continue
                exp = exposures[exp_index-1]  # select first exposure
                exp_prefixes = []
                # collect all als files for this exposure
                for prefix in prefixes:
                    if not prefix.startswith(exp):
                        continue
                    exp_prefixes.append(prefix)

                # Convert radec.orig to radec.dat, convert RA to degress
                # and add angular offset
                # mF: Not sure if we will need radec.dat later,
                # creating it for now.
                ra, dec, pa = utils.read_radec("radec.orig")
                if ra0 is not None:
                    logging.info("Overwriting RA from multifits by value "
                                 "from command line = {}".format(ra0))
                    ra = ra0
                if dec0 is not None:
                    logging.info("Overwriting DEC from multifits by value "
                                 "from command line = {}".format(dec0))
                    dec = dec0

                # Now compute offsets iteratively with increasingly
                # smaller matching radii.
                # Matching radii are defined in config file.
                ra_offset, dec_offset = 0., 0.
                for i, radius in enumerate(radii):
                    logging.info("Angular offset {:.3} Deg, getoff2 iteration"
                                 " {}, matching radius = {}\""
                                 .format(angoff, i+1, radius))
                    radec_outfile = 'tmp_exp{:02d}.csv'.format(exp_index)
                    logging.info("Adding RA & Dec to detections, "
                                 "applying offsets ra_offset,dec_offset,"
                                 "pa_offset = {},{},{}".format(ra_offset,
                                                               dec_offset,
                                                               angoff))
                    # Call add_ra_dec, add offsets first.
                    new_ra, new_dec, new_pa = ra * 15. + ra_offset, \
                        dec + dec_offset, pa + angoff
                    # New direct call to pyhetdex
                    # preload the als data.
                    fp = fplane.FPlane("fplane.txt")
                    als_files = get_als_files(fp, exp_prefixes)
                    als_data = load_als_data(als_files)
                    add_ra_dec(wdir, als_data, ra=new_ra, dec=new_dec,
                               pa=new_pa, fp=fp, radec_outfile=radec_outfile)

                    # Now compute offsets.
                    logging.info("Computing offsets ...")
                    dra_offset, ddec_offset = \
                        cltools.getoff2(radec_outfile, shout_ifustars,
                                        radius, ra_offset=0., dec_offset=0.,
                                        logging=logging)
                    ra_offset, dec_offset = \
                        ra_offset+dra_offset, dec_offset+ddec_offset
                    logging.info("End getoff2 iteration {}: Offset adjusted "
                                 "by {:.6f}, {:.6f} to {:.6f}, {:.6f}"
                                 .format(i+1, dra_offset, ddec_offset,
                                         ra_offset, dec_offset))
                    logging.info("")
                    logging.info("")

                # Copy getoff.out and getoff2.out to add_radec_angoff_trial_dir
                sangoff = '_{:06.3f}Deg'.format(angoff)
                fnout = os.path.join(add_radec_angoff_trial_dir,
                                     "getoff_exp{:02d}{}.out"
                                     .format(exp_index, sangoff))
                shutil.copy2("getoff.out", fnout)
                fnout = os.path.join(add_radec_angoff_trial_dir,
                                     "getoff2_exp{:02d}{}.out"
                                     .format(exp_index, sangoff))
                shutil.copy2("getoff2.out", fnout)

                shutil.move("getoff.out",
                            "getoff_exp{:02d}.out".format(exp_index))
                shutil.move("getoff2.out",
                            "getoff2_exp{:02d}.out".format(exp_index))
                # Write radec_XXXDeg.dat and radec2_XXXDeg.dat
                with path.Path(add_radec_angoff_trial_dir):
                    write_ra_dec_dats(ra, dec, pa, exp_index, angoff,
                                      ra_offset, dec_offset, nominal=False)
                # if the current offset angle is the nominal one,
                # then also write radec.dat and radec2.dat witouh angle
                # information in filename.
                if angoff == add_radec_angoff:
                    write_ra_dec_dats(ra, dec, pa, exp_index, angoff,
                                      ra_offset, dec_offset, nominal=True)


def combine_radec(wdir, dither_offsets, PLOT=True):
    """
    Computes - based on the RA Dec information of the individual exposures
    (from radec2_exp0?.dat) the final RA/Dec for the shot.

    Notes
    -----
        Creates radec2_final.dat.
        Optionally create a plot indicating the individual exposure positions.

    Parameters
    ----------
    wdir : str
        Work directory.
    """
    global vdrp_info
    logging.info("Combining RA, Dec positions of all exposures to "
                 "final shot RA, Dec.")
    ff = np.sort(glob.glob(wdir + "/radec2_exp??.dat"))
    ra0, dec0, pa0 = read_radec(ff[0])
    translated = []
    if PLOT:
        fig = plt.figure(figsize=[7, 7])
        # ax = plt.subplot(111)
        plt.subplot(111)
    for i, (f, offset) in enumerate(zip(ff, dither_offsets)):
        ra, dec, pa = read_radec(f)
        logging.info("Exposure {:d} RA,Dec = {:.6f},{:.6f}".format(i+1, ra,
                                                                   dec))
        rot = 360.0 - (pa + 90.)
        tp = TangentPlane(ra, dec, rot)

        xfp = 0.
        yfp = 0.
        _ra, _dec = tp.xy2raDec(xfp, yfp)

        if PLOT:
            plt.plot(ra, _dec, 's', color='grey')
            plt.text(_ra+5e-6, _dec+5e-6, i+1)

        xfp = -offset[0]
        yfp = -offset[1]
        _ra, _dec = tp.xy2raDec(xfp, yfp)

        if PLOT:
            plt.plot(_ra, _dec, 'o', color='grey')
            plt.text(_ra+5e-6, _dec+5e-6, i+1)

        translated.append([_ra, _dec])

    translated = np.array(translated)
    final_ra, final_dec = np.mean(translated[:, 0]), np.mean(translated[:, 1])
    dfinal_ra = np.std(translated[:, 0])/np.cos(np.deg2rad(dec0))
    dfinal_dec = np.std(translated[:, 1])

    s1 = "RA = {:.6f} Deg +/- {:.3f}\"".format(final_ra, dfinal_ra*3600.)
    logging.info("Final shot  " + s1)
    s2 = "Dec = {:.6f} Deg +/- {:.3f}\" ".format(final_dec, dfinal_dec*3600.)
    logging.info("Final shot  " + s2)
    if PLOT:
        plt.plot([], [], 's', color='grey', label="exposure center")
        plt.plot([], [], 'o', color='grey', label="inferred shot center")
        # l = plt.legend()
        plt.legend()
        plt.plot([final_ra], [final_dec], 'ko', markersize=10)
        plt.text(final_ra, final_dec, s1 + "\n" + s2, ha='right')
        plt.xlabel("RA [Deg]")
        plt.ylabel("Dec [Deg]")

    vdrp_info.final_ra = final_ra
    vdrp_info.final_dec = final_dec
    vdrp_info.dfinal_ra = dfinal_ra
    vdrp_info.dfinal_dec = dfinal_dec

    write_radec(final_ra, final_dec, pa0,
                os.path.join(wdir, "radec2_final.dat"))
    fig.tight_layout()
    plt.savefig(os.path.join(wdir, "radec2_final.pdf"))


def add_ifu_xy(wdir, offset_exposure_indices):
    """ Adds IFU x y information to stars used for matching,
    and save to xy_expNN.dat.
    Requires: getoff.out, radec2.dat
    Analogous to rastrom3.

    Parameters
    ----------
    wdir : str
        Work directory.
    offset_exposure_indices : list
        List of exposure indices to consider.
    """
    global vdrp_info

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
        for exp_index in offset_exposure_indices:
            logging.info("Creating xy_exp{:02d}.dat".format(exp_index))

            fngetoff_out = "getoff_exp{:02d}.out".format(exp_index)
            if not os.path.exists(fngetoff_out):
                logging.warning("Have no {} for exposure {}. Check "
                                "your configuration (offset_exposure_indices)."
                                " Skipping ..."
                                .format(fngetoff_out, exp_index))
                continue

            t = Table.read(fngetoff_out, format="ascii.fast_no_header")
            vdrp_info["getoff_nstars_exp{:02d}".format(exp_index)] = len(t)
            t_detect_coor = Table([t['col3'], t['col4'], t['col7']],
                                  names=["RA", "DEC", "IFUSLOT"])
            t_catalog_coor = Table([t['col5'], t['col6'], t['col7']],
                                   names=["RA", "DEC", "IFUSLOT"])

            # read ra,dec, pa from radec2.dat
            ra, dec, pa = utils.read_radec("radec2_exp{:02d}.dat"
                                           .format(exp_index))

            # set up astrometry
            fp = fplane.FPlane("fplane.txt")
            # Carry out required changes to astrometry
            rot = 360.0 - (pa + 90.)
            # Set up astrometry from user supplied options
            tp = TangentPlane(ra, dec, rot)

            t_detect_coor_xy = ra_dec_to_xy(t_detect_coor, ra, dec, fp, tp)
            t_catalog_coor_xy = ra_dec_to_xy(t_catalog_coor, ra, dec, fp, tp)

            renamed_t_detect_coor_xy = t_detect_coor_xy.copy()
            renamed_t_catalog_coor_xy = t_catalog_coor_xy.copy()

            for c in t_detect_coor_xy.columns:
                # t_detect_coor_xy.columns[c].name = \
                #     t_detect_coor_xy.columns[c].name + "1"
                renamed_t_detect_coor_xy.columns[c].name = \
                    t_detect_coor_xy.columns[c].name + "_det"
            for c in t_catalog_coor_xy.columns:
                # t_catalog_coor_xy.columns[c].name = \
                #     t_catalog_coor_xy.columns[c].name + "2"
                renamed_t_catalog_coor_xy.columns[c].name = \
                    t_catalog_coor_xy.columns[c].name + "_cat"

            t = table.hstack([renamed_t_detect_coor_xy,
                              renamed_t_catalog_coor_xy])
            t.write('xy_exp{:02d}.dat'.format(exp_index),
                    format="ascii.fixed_width", delimiter='', overwrite=True)


def mkmosaic(wdir, prefixes, night, shotid, mkmosaic_angoff):
    """Creates mosaic fits image.

    Parameters
    ----------
    wdir : str
        Work directory.
    prefixes : list
        List file name prefixes for the collapsed IFU images.
    night : str
        Night (e.g. 20180611)
    shotid : str
        ID of shot (e.g. 017)
    mkmosaic_angoff : float
        Angular offset to add for creation of mosaic image.
    """
    with path.Path(wdir):
        logging.info("Creating mosaic image.")
        # build mosaic from IFU images
        # exposures = np.unique([p[:15] for p in prefixes])
        # exp1 = exposures[0]
        exposures_files = get_exposures_files(".")
        for exp in exposures_files:
            logging.info("Calling immosaicv ....")
            daophot.rm(['immosaic.fits'])
            cltools.immosaicv(exposures_files[exp],
                              fplane_file="fplane.txt", logging=logging)

            # rotate mosaic to correct PA on sky
            ra, dec, pa = utils.read_radec('radec2_{}.dat'.format(exp))
            alpha = 360. - (pa + 90. + mkmosaic_angoff)

            logging.info("Calling imrot with angle {} (can take a "
                         "minute) ....".format(alpha))
            daophot.rm(['imrot.fits'])
            cltools.imrot("immosaic.fits", alpha, logging=logging)
            hdu = fits.open("imrot.fits")

            h = hdu[0].header
            h["CRVAL1"] = ra
            h["CRVAL2"] = dec
            h["CTYPE1"] = "RA---TAN"
            h["CTYPE2"] = "DEC--TAN"
            h["CD1_1"] = -0.0002777
            h["CD1_2"] = 0.
            h["CD2_2"] = 0.0002777
            h["CD2_1"] = 0
            h["CRPIX1"] = 650.0
            h["CRPIX2"] = 650.0
            h["CUNIT1"] = "deg"
            h["CUNIT2"] = "deg"
            h["EQUINOX"] = 2000

            hdu.writeto("{}v{}fp_{}.fits".format(night, shotid, exp),
                        overwrite=True)


def project_xy(wdir, radec_file, fplane_file, ra, dec):
    """Translate *all* catalog stars to x/y to display then and to
    see which ones got matched.
    Call pyhetdex tangent_plane's functionality to project
    ra,dec to x,y.

    Parameters
    ----------
    wdir : str
        Work directory.
    radec_file : str
        File that contains shot ra dec position.
    fplane_file : str
        Focal plane file filename.
    ra : list
        List of ra positions (in float, degree).
    dec : list
        List of dec positions (in float, degree).
    """
    # read ra,dec, pa from radec2.dat
    ra0, dec0, pa0 = utils.read_radec(os.path.join(wdir, radec_file))
    # Carry out required changes to astrometry
    rot = 360.0 - (pa0 + 90.)
    # Set up astrometry from user supplied options
    tp = phastrom.TangentPlane(ra0, dec0, rot)
    # set up astrometry
    fp = fplane.FPlane(os.path.join(wdir, fplane_file))
    # find positions
    ifu_xy = phastrom.ra_dec_to_xy(ra, dec, fp, tp)
    return ifu_xy


def mk_match_matrix(wdir, ax, exp, image_files, fplane_file, shout_ifu_file,
                    xy_file, radec_file):
    """ Creates the actual match plot for a specific exposures.
    This is a subroutine to mk_match_plots.

    Parameters
    ----------
    wdir : str
        Work directory.
    prefixes : list
        List file name prefixes for the collapsed IFU images.
    ax : pyplot.axes
        Axes object to plot into.
    exp : str
        Exposure string (e.g. exp01)
    image_files : list
        List of file names.
    fplane_file : str
        Focal plane file filename.
    shout_ifu_file : str
        Shuffle IFU star catalog output filename.
    xy_file : str
        Filename for list of matched stars, aka xy_exp??.dat.
    radec_file : str
        File that contains shot ra dec position.
    """
    cmap = plt.cm.bone

    N = 1.
    tin = Table.read(os.path.join(wdir, shout_ifu_file), format='ascii')
    tout = Table([tin['col2'], tin['col3'], tin['col4']],
                 names=['id', 'ra', 'dec'])

    # load images
    images = OrderedDict()
    headers = OrderedDict()
    with path.Path(wdir):
        for f in image_files:
            images[f] = fits.getdata(f + '.fits')
            headers[f] = fits.getheader(f + '.fits')

    # Here we translate *all* catalog stars to x/y to display then and to
    ifu_xy = project_xy(wdir, radec_file, fplane_file, tout['ra'], tout['dec'])
    # Read xy information, i.e. catalog derived
    # x/y positions vs. actual detecion x/y
    t = ascii.read(os.path.join(wdir, xy_file))
    matched = Table([t['IFUSLOT_cat'], t['xifu_cat'], t['yifu_cat']],
                    names=['ifuslot', 'xifu', 'yifu'])

    RMAX = 510.

    # Matrix
    ax_all = plt.axes([0., 0., 1/N, 1/N])
    # next lines only to get a legend
    ax_all.plot([], [], 'x', label="catalog", c='#2ECC71', markersize=10)
    ax_all.plot([], [], 'r+', label="detected", markersize=10)
    ax_all.plot([], [], 'o', label="matched", markersize=10,
                markerfacecolor='none', markeredgecolor='b')
    # l = ax_all.legend()
    ax_all.legend()

    ax_all.xaxis.set_visible(False)
    ax_all.yaxis.set_visible(False)

    scaling = 1.8
    s = 51. * scaling

    fp = fplane.FPlane(os.path.join(wdir, fplane_file))
    for f in images:
        ifuslot = f[-3:]

        if ifuslot not in fp.ifuslots:
            continue
        ifu = fp.by_ifuslot(ifuslot)
        x, y, xw, xy = (-(ifu.x)+RMAX-s/2)/N, (ifu.y-s/2+RMAX)/N, s/N, s/N

        ax = plt.axes([x/RMAX/2., y/RMAX/2.,
                       xw/RMAX/2., xy/RMAX/2.])

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        try:
            h = headers[f]
            xsize = h['NAXIS1']
            ysize = h['NAXIS2']
            if "CRVAL1" not in h:
                xcenter = -25.14999961853027
                logging.warning("Found no CRVAL1 in {}.fits, "
                                "using default value.".format(f))
            else:
                xcenter = h['CRVAL1']
            if "CRVAL2" not in h:
                ycenter = -25.14999961853027
                logging.warning("Found no CRVAL2 in {}.fits, "
                                "using default value.".format(f))
            else:
                ycenter = h['CRVAL2']

            extent = [0.+xcenter, xsize+xcenter,
                      0.+ycenter, ysize+ycenter]

            ax.imshow(np.rot90(images[f], k=3), extent=extent,
                      origin='lower', vmin=-5., vmax=10., cmap=cmap)

            ii = ifu_xy['ifuslot'] == int(ifuslot)
            jj = matched['ifuslot'] == int(ifuslot)

            # don't get confused about rotations
            ax.plot(- ifu_xy['yifu'][ii], ifu_xy['xifu'][ii],
                    'x', c='#2ECC71', markersize=10)
            ax.plot(- matched['yifu'][jj], matched['xifu'][jj],
                    'o', markersize=10, markerfacecolor='none',
                    markeredgecolor='b')

            ax.set_xlim([extent[0], extent[1]])
            ax.set_ylim([extent[2], extent[3]])

            # ignore if there are no stars in the IFU, EMC Added 2023-09-21
            if os.stat(os.path.join(wdir, f + '.als')).st_size > 0:
                dp = DAOPHOT_ALS.read(os.path.join(wdir, f + '.als'))
                ax.plot(-dp.data['Y']+51./2., dp.data['X']-51./2., 'r+',
                        markersize=10)
            ax.text(.975, .025, ifuslot, transform=ax.transAxes,
                    color='white', ha='right', va='bottom')
        except Exception:
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

    Parameters
    ----------
    basedir : str
        Directory to search.

    Returns
    -------
    OrderedDict : Ordered dictionary with pairs of exposure
                      string "exp??" and time and list of

    """
    ff = []
    with path.Path(basedir):
        ff = glob.glob('2???????T??????_???.fits')
    _exp_datetimes = [f[:19] for f in ff]

    exp_datetimes = np.sort(np.unique([p[:15] for p in _exp_datetimes]))

    exposures_files = OrderedDict()
    for i, edt in enumerate(exp_datetimes):
        files_for_exposure = []
        for f in ff:
            if f.startswith(edt):
                files_for_exposure.append(f.replace('.fits', ''))
        exposures_files["exp{:02d}".format(i+1)] = files_for_exposure
    return exposures_files


def mk_match_plots(wdir, prefixes):
    """Creates match plots.

    Parameters
    ----------
    wdir : str
        Work directory.
    prefixes : list
        List file name prefixes for the collapsed IFU images.
    """
    logging.info("Creating match plots.")

    shout_ifu_file = "shout.ifustars"
    exposures = ["exp01", "exp02", "exp03"]
    xy_files = {exp: "xy_{}.dat".format(exp) for exp in exposures}
    # tmp_csv_files = {exp: "tmp_{}.csv".format(exp) for exp in exposures}
    radec_files = {exp: "radec2_{}.dat".format(exp) for exp in exposures}
    fplane_file = "fplane.txt"

    with path.Path(wdir):
        exposures_files = get_exposures_files(".")
        for exp in exposures:
            f = plt.figure(figsize=[15, 15])
            ax = plt.subplot(111)
            if exp not in exposures_files:
                logging.warning("Found no image files for "
                                "exposure {}.".format(exp))
                continue
            image_files = exposures_files[exp]
            xy_file = xy_files[exp]
            radec_file = radec_files[exp]
            if os.path.exists(xy_file) and os.path.exists(radec_file):
                mk_match_matrix(wdir, ax, exp, image_files, fplane_file,
                                shout_ifu_file, xy_file, radec_file)
                f.savefig("match_{}.pdf".format(exp))


def get_prefixes(wdir):
    """
    Create list of all file prefixes based
    on the existing collapsed IFU files in the current directory.

    Parameters
    ----------
    wdir : str
        Work directory.
    """
    ff = []
    with path.Path(wdir):
        ff = glob.glob('2???????T??????_???.fits')
    return [f[:19] for f in ff]


def get_exposures(prefixes):
    """ Computes unique list of exposures from prefixes.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed configuration parameters.
    prefixes : list
        List file name prefixes for the collapsed IFU images

    Returns
    -------
        (list): Unique list of exposure strings.
    """
    return np.unique([p[:15] for p in prefixes])


def mk_fibermap_files(wdir, reduction_dir, night, shotid):
    """ Replaces ``cp_fibermap_files``. Creates ``fibermap`` files
    from multifits. These are essentially the IFUcen files in a different format.

    Parameters
    ----------
    wdir : str
        Work directory.
    reduction_dir : str
        Directory that holds panacea reductions.
    night : str
        Night (e.g. 20180611)
    shotid : str
        ID of shot (e.g. 017)
    """

    logging.info("mk_fibermap_files: Extracting fiber mapping from "
                 "multifits files.")

    with path.Path(wdir):
        # In order to gt the mapping from IFU fiber number to spectrograph,
        # amplifier and spectrum index we look at exp01. There is the built-in
        # assumption that there is always exp01 and that the mapping is
        # stample over the exposures of one shot.
        # find all mutlifits files.
        pattern = os.path.join(reduction_dir,
                               "{}/virus/virus0000{}/exp01/virus/"
                               "multi_???_???_???_??.fits".format(night,
                                                                  shotid))
        ff = glob.glob(pattern)

        # structure to store all the fibermaps
        fibermaps = OrderedDict()

        for f in ff:
            tt = f.split("_")
            ifuslot = tt[-3]
            amp = tt[-1].replace(".fits", "")

            hdu = fits.open(f)
            xy = hdu["ifupos"].data

            if ifuslot not in fibermaps:
                fibermaps[ifuslot] = []

            for i, (x, y) in enumerate(xy):
                fibermaps[ifuslot].append([x, y, amp, i+1])

        for ifuslot in fibermaps:
            t = Table(np.array(fibermaps[ifuslot]),
                      names=["XS", "YS", "amp", "mf_spec_index"])
            fnfm = "ifuslot{}.fibermap".format(ifuslot)
            t.write(fnfm, format='ascii.fixed_width', overwrite=True)


def get_fiber_coords(wdir, active_slots, dither_offsets, subdir="coords"):
    """ Calls add_ra_dec for all IFU slots and all dithers.

    The is the main routine for getcoord which computes the on-sky positions
    for all fibers.

    Essentially this is a whole bunch of calls like.:

   add_ra_dec --ftype line_detect --astrometry 262.496605 33.194212 262.975922
       --fplane /work/00115/gebhardt/maverick/sci/panacea/shifts/fplane.txt
       --ihmps 015 --fout i015_1.csv --dx 0 --dy 0 015.addin
   ...
   add_ra_dec --ftype line_detect --astrometry 262.496605 33.194212 262.975922
       --fplane /work/00115/gebhardt/maverick/sci/panacea/shifts/fplane.txt
       --ihmps 015 --fout i015_2.csv --dy 1.27 --dx -0.73 015.addin
   ...
   add_ra_dec --ftype line_detect --astrometry 262.496605 33.194212 262.975922
       --fplane /work/00115/gebhardt/maverick/sci/panacea/shifts/fplane.txt
       --ihmps 015 --fout i015_3.csv --dy 1.27 --dx 0.73 015.addin

    Notes
    -----
    This creates a list of files iIFUSLOT_DITHERNUM.csv
    that store the on-sky fiber coordinates.

    Parameters
    ----------
    wdir : str
        Work directory.

    """

    logging.info("get_fiber_coords: Computing on-sky fiber coordinates.")
    with path.Path(os.path.join(wdir, subdir)):
        ra0, dec0, pa0 = read_radec("radec2_final.dat")

        # Find which IFU slots to operate of based on the
        # existing set og *.fibermap files.
        ifuslots = []
        fibermap_files = []
        fplane = FPlane("fplane.txt")
        for slot in active_slots:
            if slot not in fplane.ifuslots:
                logging.warning("get_fiber_coords: Slot "
                                "{} not found in fplane.txt."
                                .format(slot))
                continue
            ifu = fplane.by_ifuslot(slot)

            fn = "ifuslot{}.fibermap".format(slot)
            if os.path.exists(fn):
                ifuslots.append(slot)
                fibermap_files.append(fn)
            else:
                logging.warning("get_fiber_coords: Found no fibermap file "
                                "for slot {} with IFU ID {}. This slot "
                                "delivers data however."
                                .format(slot, ifu.ifuid))

        # Carry out required changes to astrometry
        rot = 360.0 - (pa0 + 90.)

        # Set up astrometry from user supplied options
        tp = TangentPlane(ra0, dec0, rot)

        for offset_index, (dx, dy) in enumerate(dither_offsets):
            logging.info("get_fiber_coords:    offset_index {} dx "
                         "= {:.3f}, dy = {:.3f}."
                         .format(offset_index + 1, dx, dy))
            # print("ifuslots: ", ifuslots)
            # print("fibermap_files: ", fibermap_files)
            for ifuslot, fibermap_file in zip(ifuslots, fibermap_files):
                # identify ifu
                if ifuslot not in fplane.ifuslots:
                    logging.warning("IFU {} not in fplane "
                                    "file.".format(ifuslot))
                    continue
                ifu = fplane.by_ifuslot(ifuslot)

                # read fiber positions in IFU system
                fm = ascii.read(fibermap_file, format="fixed_width")
                x, y = fm["XS"], fm["YS"]
                # skip empty tables
                if len(x) < 1:
                    continue
                # remember to flip x,y
                xfp = x + ifu.y + dx
                yfp = y + ifu.x + dy
                # project to sky
                # print("ifuslot, fibermap_file, ifu.x, dx, ifu.y , dy, ra0, "
                #       "dec0, pa0", ifuslot, fibermap_file, ifu.x, dx, ifu.y ,
                #       dy, ra0, dec0, pa0)

                ra, dec = tp.xy2raDec(xfp, yfp)
                # save results
                # construct new table that will hole fiber positions in
                # focal plane system
                cxs = Column(name="XS", data=x, dtype=float)
                cys = Column(name="YS", data=y, dtype=float)
                cra = Column(name="ra", data=ra, dtype=float)
                cdec = Column(name="dec", data=dec, dtype=float)
                cifuslot = Column(name="ifuslot", data=[ifuslot]*len(ra),
                                  dtype="S3")
                cxfplane = Column(name="xfplane", data=xfp, dtype=float)
                cyfplane = Column(name="yfplane", data=yfp, dtype=float)
                camp = Column(name="amp", data=fm["amp"], dtype='S2')
                cmf_spec_index = Column(name="mf_spec_index",
                                        data=fm["mf_spec_index"],
                                        dtype=int)
                table = Table([cxs, cys, cra, cdec, cifuslot, cxfplane,
                               cyfplane, camp, cmf_spec_index])

                outfilename = "i{}_{}.csv".format(ifuslot, offset_index + 1)
                logging.info("Writing {}.".format(outfilename))
                table.write(outfilename, comment='#', format='ascii.csv',
                            overwrite=True)


def get_active_slots(wdir, reduction_dir, exposures, night, shotid):
    """
    Figures out which IFU slots actually delivered data, by checking
    if a corresponding multifits exists for all exposures in a shot.
    """
    with path.Path(wdir):
        # will hold unique list of all slots that delelivered
        # data in any exposure
        all_slots = []
        # will hold unique lists of all slots that delelivered
        # data for this exposure
        exp_slots = {}
        for exp in exposures:
            pattern = os.path.join(reduction_dir,
                                   "{}/virus/virus0000{}/*/*/"
                                   "multi*".format(night, shotid))
            ff = glob.glob(pattern)
            exp_slots[exp] = []
            for f in ff:
                # could get ifuslot from filename
                # h = fits.getheader(f)
                # ifuslot = h["IFUSLOT"]
                # doing it from the filename is much faster
                __, t = os.path.split(f)
                ifuslot = t[10:13]
                exp_slots[exp].append(ifuslot)

            exp_slots[exp] = list(np.sort(np.unique(exp_slots[exp])))
            all_slots += list(exp_slots[exp])
        all_slots = np.sort(np.unique(all_slots))

        # Now see if all slots that have data in any exposure
        # also have data in all exposures
        final_slots = []
        for slot in all_slots:
            has_all = True
            for exp in exposures:
                if slot not in exp_slots[exp]:
                    has_all = False
                    logging.warning("Slot has some data in other "
                                    "exposures, but not in {}.".format(exp))
            if has_all:
                final_slots.append(slot)
        return final_slots


def comp_multifits(ifuslot, ifuid, specid, amp, index):
    """
    Computes multifits file links from
    ifuslot, ifuid, specid
    and lists of
    amplifier and index
    called by mk_dithall.
    """
    mf = []
    for a, i in zip(amp, index):
        fname = "multi_{specid:03d}_{ifuslot:03d}_{ifuid:03d}_{amp}_" \
            "{index:03d}.ixy"
        mf.append(fname.format(specid=int(specid), ifuslot=int(ifuslot),
                               ifuid=int(ifuid), amp=a, index=i))
    return mf


def mk_dithall(wdir, active_slots, reduction_dir, night, shotid, subdir="."):
    """
    This creates the dithall.use file that is required by the downstream
    processing functions like photometry and detect.

    The file dithall.use contains for every exposure (1-3) and every fiber the
    RA/Dec on sky coordinats and the multifits file where the spectrum is
    stored and the fiber number.
    """
    logging.info("get_fiber_coords: Computing on-sky fiber coordinates.")
    with path.Path(os.path.join(wdir, subdir)):
        # get sorted list of IFU slots from fplane file
        elist = OrderedDict()
        pattern = os.path.join(reduction_dir,
                               "{}/virus/virus0000{}/exp*"
                               .format(night, shotid))
        ee = glob.glob(pattern)
        exposures = []
        for e in ee:
            __, t = os.path.split(e)
            exposures.append(t)
        exposures = np.sort(exposures)
        for exp in exposures:
            pattern = os.path.join(reduction_dir,
                                   "{}/virus/virus0000{}/{}/virus/Co*"
                                   .format(night, shotid, exp))
            logging.info("    using Co*fits files {}..."
                         .format(pattern))
            CoFeS = glob.glob(pattern)
            __, t = os.path.split(CoFeS[0])
            prefix = t[5:22]
            elist[exp] = prefix

        column_names = "ra", "dec", "ifuslot", "XS", "YS", "xfplane", \
            "yfplane", "multifits", "timestamp", "exposure"
        # read list of exposures
        all_tdith = []

        # need fplane to translate ifuslot to ifu id
        fp = fplane.FPlane("fplane.txt")
        for i, exp in enumerate(exposures):
            logging.info("get_fiber_coords: Exposure {} ...".format(exp))
            exp_tdith = []
            for ifuslot in active_slots:
                if ifuslot not in fp.ifuslots:
                    logging.warning("mk_dithall: IFU slot {} not"
                                    " in fplane.txt. Please update fplane.txt."
                                    .format(ifuslot))
                    continue
                ifu = fp.by_ifuslot(ifuslot)

                # read the fibermaps, those contain the mapping of
                # x/y (IFU space) to fiber number on the detector
                # fibermap_filename = "ifuid{}.fibermap".format(ifu.ifuid)
                # if not os.path.exists(fibermap_filename):
                #    logging.warning("mk_dithall: Found no fibermap file for "
                #                    "IFU slot {} and IFU ID {} (expected {})."
                #                    .format(ifuslot, ifu.ifuid,
                #                            fibermap_filename))
                #    continue

                # fibermap = ascii.read(fibermap_filename)
                csv_filename = "i{}_{}.csv".format(ifuslot, i+1)
                if not os.path.exists(csv_filename):
                    logging.warning("mk_dithall: Found no *.csv file for "
                                    "IFU slot {} (expected {})."
                                    .format(ifuslot, csv_filename))
                    continue
                csv = ascii.read(csv_filename)

                # pointers to the multi extention fits and fiber
                # strings like: multi_301_015_038_RU_085.ixy
                cmulti_name = comp_multifits(ifuslot, ifu.ifuid, ifu.specid,
                                             csv["amp"], csv["mf_spec_index"])

                cifu = Column(["ifu{}".format(ifuslot)] * len(csv))

                cprefix = Column([elist[exp]] * len(csv))
                cexp = Column([exp] * len(csv))
                cc = csv["ra"], csv["dec"], cifu, csv["XS"], csv["YS"], \
                    csv["xfplane"], csv["yfplane"], cmulti_name, cprefix, cexp
                tdith = Table(cc, names=column_names)
                # try to match Karl's formatting
                tdith["ra"].format = "%10.7f"
                tdith["dec"].format = "%10.7f"
                tdith["ifuslot"].format = "%6s"
                tdith["XS"].format = "%8.3f"
                tdith["YS"].format = "%8.3f"
                tdith["xfplane"].format = "%8.3f"
                tdith["yfplane"].format = "%8.3f"
                tdith["multifits"].format = "%28s"
                tdith["timestamp"].format = "%17s"
                tdith["exposure"].format = "%5s"

                all_tdith.append(tdith)
                exp_tdith.append(tdith)
            vstack(exp_tdith).write("dith_{}.all".format(exp), overwrite=True,
                                    format='ascii.fixed_width', delimiter="")

        tall_tdithx = vstack(all_tdith)
        tall_tdithx.write("dithall.use", overwrite=True,
                          format='ascii.fixed_width', delimiter="")


def cp_results(tmp_dir, results_dir):
    """ Copies all relevant result files
    from tmp_dir results_dir.

    Parameters
    ----------
    tmp_dir : str
        Temporary work directory.
    results_dir : str
        Final directory for results.

    """
    dirs = ['add_radec_angoff_trial']
    file_pattern = []
#    file_pattern += ["CoFeS*_???_sci.fits"]
#    file_pattern += ["*.als"]
    file_pattern += ["*tot.als"]
#    file_pattern += ["*.ap"]
#    file_pattern += ["*.coo"]
#    file_pattern += ["*.lst"]
    file_pattern += ["2*fp_exp??.fits"]
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
    file_pattern += ["shout.*"]
#    file_pattern += ["shout.info"]
#    file_pattern += ["shout.probestars"]
#    file_pattern += ["shout.result"]
    file_pattern += ["shuffle.cfg"]
#    file_pattern += ["tmp_exp??.csv"]
    file_pattern += ["use.psf"]
    file_pattern += ["2*fp.fits"]
    file_pattern += ["xy_exp??.dat"]
    file_pattern += ["match_*.pdf"]
    file_pattern += ["radec2_final.dat"]
    file_pattern += ["radec2_final.pdf"]
    file_pattern += ["vdrp_info.pickle"]
    file_pattern += ["dithall.use"]
    file_pattern += ["dith_exp??.all"]

    for d in dirs:
        td = os.path.join(tmp_dir, d)
        if os.path.exists(td):
            dir_util.copy_tree(td, os.path.join(results_dir, d))
    for p in file_pattern:
        ff = glob.glob("{}/{}".format(tmp_dir, p))
        for f in ff:
            shutil.copy2(f, results_dir)


vdrp_info = None


def main(args):
    """
    Main function.
    """
    global vdrp_info

    # Create results directory for given night and shot
    cwd = os.getcwd()
    results_dir = os.path.join(cwd, "{}v{}".format(args.night, args.shotid))
    utils.createDir(results_dir)

    fmt = '%(asctime)s %(levelname)-8s %(funcName)15s(): %(message)s'
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(results_dir, args.logfile),
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter(fmt)
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # save arguments for the execution
    with open(os.path.join(results_dir, 'args.pickle'), 'wb') as f:
        pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)

    tasks = args.task.split(",")
    if args.use_tmp and not tasks == ['all']:
        logging.error("Step-by-step execution not possile when running "
                      "in a tmp directory.")
        logging.error("   Please either call without -t or set "
                      "use_tmp to False.")
        sys.exit(1)
    logging.info("Executing tasks : {}".format(tasks))

    # default is to work in results_dir
    wdir = results_dir
    if args.use_tmp:
        # Create a temporary directory
        tmp_dir = tempfile.mkdtemp(dir=args.tmp_dir)
        logging.info("Tempdir is {}".format(tmp_dir))
        logging.info("Copying over prior data (if any)...")
        dir_util.copy_tree(results_dir, tmp_dir)
        # set working directory to tmp_dir
        wdir = tmp_dir

    logging.info("Configuration {}.".format(args.config_source))

    vdrp_info = VdrpInfo.read(wdir)
    vdrp_info.night = args.night
    vdrp_info.shotid = args.shotid

    try:
        for task in tasks:
            if task in ["cp_post_stamps", "all"]:
                # Copy over collapsed IFU cubes, aka IFU postage stamps.
                cp_post_stamps(wdir, args.reduction_dir, args.night,
                               args.shotid)

            prefixes = get_prefixes(wdir)
            exposures = get_exposures(prefixes)

            if task in ["mk_post_stamp_matrix", "all"]:
                # Create IFU postage stamp matrix image.
                mk_post_stamp_matrix(wdir, prefixes, args.cofes_vis_vmin,
                                     args.cofes_vis_vmax)

            if task in ["daophot_find", "all"]:
                # Run initial object detection in postage stamps.
                daophot_find(wdir, prefixes, args.daophot_opt,
                             args.daophot_sigma, args.daophot_xmin,
                             args.daophot_xmax, args.daophot_ymin,
                             args.daophot_ymix)

            if task in ["daophot_phot_and_allstar", "all"]:
                # Run photometry
                daophot_phot_and_allstar(wdir, prefixes,
                                         args.daophot_photo_opt,
                                         args.daophot_allstar_opt,
                                         args.daophot_phot_psf)

            if task in ["mktot", "all"]:
                # Combine detections accross all IFUs.
                mktot(wdir, prefixes, args.mktot_ifu_grid, args.mktot_magmin,
                      args.mktot_magmax, args.mktot_xmin, args.mktot_xmax,
                      args.mktot_ymin, args.mktot_ymax, args.dither_offsets)

            if task in ["rmaster", "all"]:
                # Run daophot master to ???
                if len(exposures) > 1:
                    rmaster(wdir)
                else:
                    logging.info("Only one exposure, skipping rmaster.")

            if task in ["flux_norm", "all"]:
                # Compute relative flux normalisation.
                if len(exposures) > 1:
                    flux_norm(wdir, args.fluxnorm_mag_max)
                else:
                    logging.info("Only one exposure, skipping flux_norm.")

            if task in ["get_ra_dec_orig", "all"]:
                # Retrieve original RA DEC from one of the multi files.
                # store in radec.orig
                get_ra_dec_orig(wdir, args.reduction_dir, args.night,
                                args.shotid, args.parangle)

            if task in ["redo_shuffle", "all"]:
                # Rerun shuffle to get IFU stars
                # if ra,dec, track were passed on command line, then use those
                # otherwise use corresponding vaules from the multifits header.
                # Note that get_ra_dec_orig read the multifits and stores
                # the ra,dec in radec.orig.
                ra, dec, track = args.ra, args.dec, args.track
                if track is None:
                    track = get_track(wdir, args.reduction_dir, args.night,
                                      args.shotid)
                if ra is None or dec is None:
                    _ra, _dec, _pa = \
                        utils.read_radec(os.path.join(wdir, "radec.orig"))
                    if ra is None:
                        ra = _ra
                    if dec is None:
                        dec = _dec
                # First make sure we run with SDSS to get the stars list
                redo_shuffle(wdir, ra, dec, track,
                             args.acam_magadd, args.wfs1_magadd,
                             args.wfs2_magadd, args.shuffle_cfg,
                             args.fplane_txt, args.night)
                #             args.fplane_txt, args.night, catalog='SDSS')
                if False:
                    # Now try to run it using GAIA, for the astrometry
                    logging.info('Trying shuffle with GAIA')
                    redo_shuffle(wdir, ra, dec, track,
                                 args.acam_magadd, args.wfs1_magadd,
                                 args.wfs2_magadd, args.shuffle_cfg,
                                 args.fplane_txt, args.night, catalog='GAIA')
                    with path.Path(wdir):
                        # check the number of stars in the shout.ifustars
                        if utils.count_lines('shout.ifustars') < 2:
                            # No stars found, check SDSS results
                            logging.info('No GAIA stars found, checking SDSS')
                            if utils.count_lines('sdds.ifustars') > 1:
                                # SDSS stars found, use these:
                                shutil.copy('sdss.ifustars', 'shout.ifustars')
                            else:
                                logging.info('No SDSS stars found, '
                                             'checking USNO')
                                # Finally use USNO as last fallback
                                redo_shuffle(wdir, ra, dec, track,
                                             args.acam_magadd,
                                             args.wfs1_magadd,
                                             args.wfs2_magadd,
                                             args.shuffle_cfg,
                                             args.fplane_txt, args.night,
                                             catalog='USNO')

            if task in ["compute_offset", "all"]:
                # Compute offsets by matching
                # detected stars to sdss stars from shuffle.
                # This also calls add_ra_dec to add RA DEC
                # information to detections.
                compute_offset(wdir, prefixes, args.getoff2_radii,
                               args.add_radec_angoff_trial,
                               args.add_radec_angoff,
                               args.add_radec_angoff_trial_dir,
                               args.offset_exposure_indices,
                               final_ang_offset=None,
                               shout_ifustars='shout.ifustars',
                               ra0=args.ra, dec0=args.dec)

            if task in ["compute_with_optimal_ang_off", "all"]:
                # Compute offsets by matching
                trial_dir = os.path.join(wdir, "add_radec_angoff_trial")
                optimal_ang_off = \
                    compute_optimal_ang_off(trial_dir,
                                            smoothing=args.optimal_ang_off_smoothing,
                                            PLOT=True)
                compute_offset(wdir, prefixes, args.getoff2_radii,
                               args.add_radec_angoff_trial,
                               args.add_radec_angoff,
                               args.add_radec_angoff_trial_dir,
                               args.offset_exposure_indices,
                               final_ang_offset=optimal_ang_off,
                               shout_ifustars='shout.ifustars',
                               ra0=args.ra, dec0=args.dec)

            if task in ["combine_radec", "all"]:
                # Combine individual exposure radec information.
                combine_radec(wdir, args.dither_offsets)

            if task in ["add_ifu_xy", "all"]:
                add_ifu_xy(wdir, args.offset_exposure_indices)

            if task in ["mkmosaic", "all"]:
                # build mosaic for focal plane
                mkmosaic(wdir, prefixes, args.night, args.shotid,
                         args.mkmosaic_angoff)

            if task in ["mk_match_plots", "all"]:
                # build mosaic for focal plane
                mk_match_plots(wdir, prefixes)

            if task in ["fibcoords", "all"]:

                # Create `fibermap` files. These contain the IFU x/y to fiber
                # number mapping.
                mk_fibermap_files(wdir, args.reduction_dir, args.night,
                                  args.shotid)

                # find which slots delivered data for all exposures
                # (infer from existance of corresponding multifits files).
                active_slots = get_active_slots(wdir, args.reduction_dir,
                                                exposures, args.night,
                                                args.shotid)
                logging.info("Found following exposures: {}".format(exposures))
                logging.info("Found {} active slots.".format(len(active_slots)))

                # This is where the main work happens.
                # Essentially calls add_ra_dec for all IFU slots
                # and all dithers.
                # Actually it uses the direct python interface to tangent_plane
                # rhater than calling add_ra_dec.
                get_fiber_coords(wdir, active_slots, args.dither_offsets,
                                 subdir=".")

                # Create final dithall.use file for downstream functions.
                mk_dithall(wdir, active_slots, args.reduction_dir, args.night,
                           args.shotid, subdir=".")

    finally:
        vdrp_info.save(wdir)
        if args.use_tmp:
            logging.info("Copying over results.")
            cp_results(tmp_dir, results_dir)
            if args.remove_tmp:
                logging.info("Removing temporary directoy.")
                shutil.rmtree(tmp_dir)
        logging.info("Done.")


def run():
    argv = None
    if argv is None:
        argv = sys.argv
    # Parse config file and command line paramters
    # command line parameters overwrite config file.
    args = parseArgs(argv)
    sys.exit(main(args))


if __name__ == "__main__":
    run()
