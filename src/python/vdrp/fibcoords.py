#!/usr/bin/env python
""" Astrometry routine

Module to add astrometry to HETDEX catalgoues and images
Contains python translation of Karl Gebhardt

.. moduleauthor:: Maximilian Fabricius <mxhf@mpe.mpg.de>
"""
from __future__ import print_function

import numpy as np
from numpy import loadtxt
import argparse
import os
import glob
import shutil
import sys
import ConfigParser
import logging
import subprocess
from collections import OrderedDict
import tempfile
import path
from distutils import dir_util

from astropy.io import fits
from astropy.table import Table
from astropy import table
from astropy.table import Table
from astropy.stats import biweight_location as biwgt_loc

from pyhetdex.het import fplane
from pyhetdex.het.fplane import FPlane
from pyhetdex.coordinates.tangent_projection import TangentPlane
import pyhetdex.tools.read_catalogues as rc

from vdrp.cofes_vis import cofes_4x4_plots
from vdrp import daophot
from vdrp import cltools
from vdrp.utils import createDir
from vdrp.utils import read_radec
from vdrp.utils import read_all_mch

from vdrp.utils import rm


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
    defaults["logfile"] = "astrometry.log"
    defaults["reduction_dir"] = "reductions/"
    defaults["addin_dir"] = "vdrp/config/"
    defaults["shifts_dir"] = "shifts/"

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
    parser.add_argument("--logfile", type=str)
    parser.add_argument("--addin_dir", type=str)
    parser.add_argument("--shifts_dir", type=str)

    # positional arguments
    parser.add_argument('night', metavar='night', type=str,
            help='Night of observation (e.g. 20180611).')
    parser.add_argument('shotid', metavar='shotid', type=str,
            help='Shot ID (e.g. 017).')

    args = parser.parse_args(remaining_argv)



    return args


def get_exposures(args):
    """
    Search reductions directory and find how many exposures there are.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.

    Returns:
        (list): Sorted list with exposures in reduction directory for night and shot
                (from args). E.g. ['exp01', exp02', 'exp03'].
    """
    pattern = os.path.join( args.reduction_dir, "{}/virus/virus0000{}/exp??".format( args.night, args.shotid ) )
    expdirs = glob.glob(pattern)
    exposures = []
    for d in expdirs:
        __, t = os.path.split(d)
        exposures.append(t)

    return np.sort( exposures )


def mk_exp_sub_dirs(args, wdir, exposures):
    """ Make subdirectory structure.
    Creates ./exp??/virus

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
        exposures (list): Sorted list with exposures in reduction directory for night and shot
                (from args). E.g. ['exp01', exp02', 'exp03'].

    """
    logging.info("mk_exp_sub_dirs: Creating exp0?/virus structure.")
    with path.Path(wdir):
        for exp in exposures:
            createDir(exp + "/virus")


def mk_coords_sub_dir(args, wdir):
    """ Make subdirectory structure.
    Creates ./coords

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
        exposures (list): Sorted list with exposures in reduction directory for night and shot
                (from args). E.g. ['exp01', exp02', 'exp03'].

    """
    logging.info("mk_coords_sub_dir: Creating coords subdirectory.")
    with path.Path(wdir):
        createDir("coords")



def link_multifits(args, wdir, exposures):
    """ Link Panacea's multifits files into exp??/virus subdiretories.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
        exposures (list): Sorted list with exposures in reduction directory for night and shot
                (from args). E.g. ['exp01', exp02', 'exp03'].

    """
    logging.info("link_multifits: Creating links to multi*fits files.")
    with path.Path(wdir):
        for exp in exposures:
            pattern = os.path.join( args.reduction_dir, "{}/virus/virus0000{}/{}/virus/multi*"\
                    .format( args.night, args.shotid, exp ) )
            logging.info("    Creating links to multi*fits files {}...".format(pattern))
            multifits = glob.glob(pattern)
            logging.info("    Linking {} files ...".format(len(multifits)))
            for mf in multifits:
                __,t = os.path.split(mf)
                target = os.path.join(wdir, "{}/virus/{}".format(exp, t))
                rm([target])
                os.symlink(mf,os.path.join(wdir, target))


def cp_astrometry(args, wdir):
    """ Copies astrometry information from
    shifts directory.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
    """
    logging.info("cp_astrometry: Copy over all.mch and radec2.dat from {}.".format(args.shifts_dir))
    ff = []
    ff.append( "{}/{}v{}/all.mch".format(args.shifts_dir, args.night, args.shotid) )
    ff.append(  "{}/{}v{}/radec2.dat".format(args.shifts_dir, args.night, args.shotid) )
    for f in ff:
        shutil.copy2(f, os.path.join(wdir,"coords") )


def cp_addin_files(args, wdir):
    """ Copies `addin` files. These are
    essentially the IFUcen files in a different format.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
    """
    logging.info("cp_addin_files: Copy over *.addin from {}.".format(args.addin_dir))
    pattern = args.addin_dir + "/*.addin"
    ff = glob.glob(pattern)
    logging.info("    Found {} files.".format(len(ff)))
    for f in ff:
        shutil.copy2(f, os.path.join(wdir,"coords") )


def get_fiber_coords(args, wdir):
    """ Calls add_ra_dec for all IFU slots and all dithers.

    The is the main routine for getcoord which computes the on-sky positions
    for all fibers.

    Essentially this is a whole bunch of calls like.

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
    ...

    Notes:
        This creates a list of files iIFUSLOT_DITHERNUM.csv that store the on-sky
        fiber coordinates.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.

    """

    logging.info("get_fiber_coords: Computing on-sky fiber coordinates.")
    with path.Path(os.path.join(wdir, "coords")):
        ra0, dec0, pa0 = read_radec("radec2.dat")
        dither_offsets = read_all_mch("all.mch")

        # Find which IFU slots to operate of based on the
        # existing set og *.addin files.
        addin_files = glob.glob("???.addin")
        ifuslots = []
        for f in addin_files:
            ifuslots.append( f[:3] )

        # Carry out required changes to astrometry
        rot = 360.0 - (pa0 + 90.)

        # Set up astrometry from user supplied options
        tp = TangentPlane(ra0, dec0, rot)

        fplane = FPlane(args.fplane_txt)
        for offset_index in dither_offsets:
            for ifuslot,addin_file in zip(ifuslots,addin_files):
                # idetify ifu 
                ifu = fplane.by_ifuslot(ifuslot)
                dx, dy = dither_offsets[offset_index]
                # remember to flip x,y
                xfp = ifu.y + dx
                yfp = ifu.x + dy
                # read fiber positions 
                x, y, table = rc.read_line_detect(addin_file)
                # skip empty tables
                if len(x) < 1:
                    continue
                # remember to flip x,y
                xfp = x + ifu.y + dx
                yfp = y + ifu.x + dy
                # project to sky
                ra, dec= tp.xy2raDec(xfp, yfp)
                # save results
                table['ra'] = ra
                table['dec'] = dec
                table['ifuslot'] = ifuslot
                table['xfplane'] = xfp
                table['yfplane'] = yfp
                outfilename = "i{}_{}.csv".format(ifuslot, offset_index)
                table.write(outfilename, comment='#', format='ascii.csv', overwrite=True)


def config_logger(args):
    """ Setup logging to file and screen.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
    """
    # set up logging to file
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


def main():
    """
    Main function.
    """
    # Parse config file and command line paramters
    # command line parameters overwrite config file.
    args = parseArgs()

    # Set up logging
    config_logger(args)

    logging.info("Start.")

    # Create results directory for given night and shot
    cwd = os.getcwd()
    wdir = os.path.join(cwd, "{}v{}".format(args.night, args.shotid))
    createDir(wdir)

    exposures = get_exposures(args)

    # create subdirectories for each exposure and ./coords
    mk_exp_sub_dirs(args, wdir, exposures)
    mk_coords_sub_dir(args, wdir)

    # create symlinks to multi*fits fitsfiles
    link_multifits(args, wdir, exposures)

    # copy astrometry solution (all.mch and radec2.dat) from shifts directory
    cp_astrometry(args, wdir)

    # Copies `addin` files. These are essentially the IFUcen files in a different format.
    cp_addin_files(args, wdir)

    # The is where the main work happens.
    # Essentially calls add_ra_dec for all IFU slots and all dithers.
    # Actually it uses the direct python interface to tangent_plane thater than calling add_ra_dec.
    get_fiber_coords(args, wdir)

    logging.info("Done.")

    sys.exit(0)


if __name__ == "__main__":
    sys.exit(main())
