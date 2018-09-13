#!/usr/bin/env python
""" Astrometry routine

Module to add astrometry to HETDEX catalgoues and images
Contains python translation of Karl Gebhardt

.. moduleauthor:: Maximilian Fabricius <mxhf@mpe.mpg.de>
"""

from __future__ import print_function
import matplotlib

from matplotlib import pyplot as plt

from numpy import loadtxt
from argparse import RawDescriptionHelpFormatter as ap_RDHF
from argparse import ArgumentParser as AP

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
import pickle
import ast

# import scipy
from scipy.interpolate import UnivariateSpline

from distutils import dir_util

import path
from astropy import table
from astropy.table import Table

from astropy.stats import biweight_location as biwgt_loc
from astropy.table import vstack

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

matplotlib.use("agg")


class VdrpInfo(OrderedDict):
    def __init__(self, *args, **kwargs):
        super(VdrpInfo, self).__init__(*args, **kwargs)

    def save(self, dir, filename='vdrp_info.pickle'):
        # save arguments for the execution
        with open(os.path.join(dir, filename), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def read(dir, filename='vdrp_info.pickle'):
        if os.path.exists(os.path.join(dir, filename)):
            with open(os.path.join(dir, filename), 'rb') as f:
                return pickle.load(f)
        else:
            return VdrpInfo()


def parseArgs(args):
    """ Parses configuration file and command line arguments.
    Command line arguments overwrite configuration file settiongs which
    in turn overwrite default values.

    Args:
        args (argparse.Namespace): Return the populated namespace.
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

    defaults = {}
    defaults["use_tmp"] = "False"
    defaults["remove_tmp"] = "True"

    config_source = "Default"
    if args.conf_file:
        config_source = args.conf_file
        config = ConfigParser.SafeConfigParser()
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

    # positional arguments
    # parser.add_argument('night', metavar='night', type=str,
    #                     help='Night of observation (e.g. 20180611).')
    # parser.add_argument('shotid', metavar='shotid', type=str,
    #                     help='Shot ID (e.g. 017).')
    # parser.add_argument('ra', metavar='ra', type=float,
    #                     help='RA of the target in decimal hours.')
    # parser.add_argument('dec', metavar='dec', type=float,
    #                     help='Dec of the target in decimal hours degree.')
    # parser.add_argument('track', metavar='track', type=int, choices=[0, 1],
    #                     help='Type of track: 0: East 1: West')

    args = parser.parse_args(remaining_argv)

    args.config_source = config_source
    # should in principle be able to do this with accumulate???
    args.use_tmp = args.use_tmp == "True"
    args.remove_tmp = args.remove_tmp == "True"

    return args


def prepare_model():
    """
    Corresponds to rsp2 script
    """

    pass


def average_spectrum(wl, cnts, amp_norm, tp_norm, wlmin, wlmax):
    """
    Corresponds to ravgsp0 script

    Parameters
    ----------
    wl : ndarray
        Spectrum wavelengths
    cnts : ndarray
        Counts of spectrum at the wavelength points
    amp_norm : ndarray

    tp_norm : ndarray

    wlmin : float
        Minimum wavelength to average
    wlmax : float
        Maximum  wavelength to average

    """

    wh = (wl > wlmin) & (wl < wlmax)

    # Calculate the mean of all values within wavelength range
    # where cnts are !=0

    avg = cnts[wh][cnts != 0.].mean()
    norm = (amp_norm[wh]*tp_norm[wh]).sum()

    return avg, norm


def cp_results(tmp_dir, results_dir):
    """ Copies all relevant result files
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

    logging.info("Configuration {}.".format(args.config_source))

    vdrp_info = VdrpInfo.read(wdir)
    vdrp_info.night = args.night
    vdrp_info.shotid = args.shotid

    try:
        for task in tasks:
            pass
            # if task in ["cp_post_stamps", "all"]:
            #    # Copy over collapsed IFU cubes, aka IFU postage stamps.
            #    cp_post_stamps(wdir, args.reduction_dir, args.night,
            #                   args.shotid)


    finally:
        vdrp_info.save(wdir)
        if args.use_tmp:
            logging.info("Copying over results.")
            cp_results(tmp_dir, results_dir)
            if args.remove_tmp:
                logging.info("Removing temporary directoy.")
                shutil.rmtree(tmp_dir)
        logging.info("Done.")


if __name__ == "__main__":
    argv = None
    if argv is None:
        argv = sys.argv
    # Parse config file and command line paramters
    # command line parameters overwrite config file.
    args = parseArgs(argv)

    sys.exit(main(args))
