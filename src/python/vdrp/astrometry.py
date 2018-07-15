import numpy as np
import argparse
import os
import glob
import shutil
import sys
import ConfigParser
from vdrp.cofes_vis import cofes_4x4_plots
from vdrp import daophot
import path

def parseArgs():
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
    defaults["option"] = "default"
    defaults["reduction_dir"] = "/work/04287/mxhf/maverick/red1/reductions/"
    defaults["cofes_vis_vmin"] = -15.
    defaults["cofes_vis_vmax"] = 25.
    defaults["daophot_sigma"]  = 2
    defaults["daophot_opt_VAR"] = 2
    defaults["daophot_opt_READ"] = 1.06
    defaults["daophot_opt_LOW"] = 10
    defaults["daophot_opt_FWHM"] = 2.0
    defaults["daophot_opt_WATCH"] = -1
    defaults["daophot_opt_PSF"] = 7.
    defaults["daophot_opt_GAIN"] = 1.274
    defaults["daophot_opt_HIGH"] = 84000.
    defaults["daophot_opt_THRESHOLD"] = 15.
    defaults["daophot_opt_FIT"] = 4.0
    defaults["daophot_opt_EX"] = 5
    defaults["daophot_opt_AN"] = 1
    defaults["daophot_xmin"] = 4
    defaults["daophot_xmax"] = 45
    defaults["daophot_ymin"] = 4
    defaults["daophot_ymix"] = 45
    defaults["daophot_phot_psf"]    = "vdrp/config/use.psf"
    defaults["daophot_photo_opt"]   = "vdrp/config/photo.opt"
    defaults["daophot_allstar_opt"] = "vdrp/config/allstar.opt"

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
    parser.add_argument("--option")
    parser.add_argument("--reduction_dir")
    parser.add_argument("--cofes_vis_vmin", type=float)
    parser.add_argument("--cofes_vis_vmax", type=float)
    parser.add_argument("--daophot_sigma",  type=float)
    parser.add_argument("--daophot_opt_VAR",  type=float)
    parser.add_argument("--daophot_opt_READ",  type=float)
    parser.add_argument("--daophot_opt_LOW",  type=float)
    parser.add_argument("--daophot_opt_FWHM",  type=float)
    parser.add_argument("--daophot_opt_WATCH",  type=float)
    parser.add_argument("--daophot_opt_PSF",  type=float)
    parser.add_argument("--daophot_opt_GAIN",  type=float)
    parser.add_argument("--daophot_opt_HIGH",  type=float)
    parser.add_argument("--daophot_opt_THRESHOLD",  type=float)
    parser.add_argument("--daophot_opt_FIT",  type=float)
    parser.add_argument("--daophot_opt_EX",  type=float)
    parser.add_argument("--daophot_opt_AN",  type=float)
    parser.add_argument("--daophot_xmin",  type=float)
    parser.add_argument("--daophot_xmax",  type=float)
    parser.add_argument("--daophot_ymin",  type=float)
    parser.add_argument("--daophot_ymix",  type=float)
    parser.add_argument("--daophot_phot_psf",  type=str)
    parser.add_argument("--daophot_photo_opt",  type=str)
    parser.add_argument("--daophot_allstar_opt",  type=str)

    parser.add_argument('night', metavar='night', type=str,
            help='Night of observation (e.g. 20180611).')
    parser.add_argument('shotid', metavar='shotid', type=str,
            help='Shot ID (e.g. 017).')
    args = parser.parse_args(remaining_argv)
 
    return args


def createDir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

args = parseArgs()

# create work directory for given night and shot
cwd = os.getcwd()
wdir = os.path.join(cwd, "{}v{}".format(args.night, args.shotid))
createDir(wdir)

def copy_postage_stamps(args):
    ## find the IFU postage stamp fits files and copy them over
    pattern = os.path.join( args.reduction_dir, "{}/virus/virus0000{}/*/*/CoFeS*".format( args.night, args.shotid ) )
    print("Copy CoFeS* files to {}".format(wdir))
    cofes_files = glob.glob(pattern)

    already_warned = False
    for f in cofes_files:
        h,t = os.path.split(f)
        if os.path.exists(os.path.join(wdir,t)):
            if not already_warned:
                print("{} already exists in {}, skipping, wo'nt warn about other files....".format(t,wdir))
                already_warned = True
            continue
        shutil.copy2(f, wdir)
    return cofes_files

def create_postage_stamp_matrix(args, cofes_files):
    # create the IFU postage stamp matrix image
    print("Creating the IFU postage stamp matrix image...")
    prefix = os.path.split(cofes_files[0])[-1][:20]
    outfile_name = prefix + ".png"
    with path.Path(wdir):
        #os.chdir(wdir)
        print(prefix, outfile_name, args.cofes_vis_vmin, args.cofes_vis_vmax)
        cofes_4x4_plots(prefix = prefix, outfile_name = outfile_name, vmin = args.cofes_vis_vmin, vmax = args.cofes_vis_vmax)


def inital_daophot_find(args, cofes_files):
    # run initial daophot find
    print("Running initial daophot find...")
    # Create configuration file for daophot.
    daophot.mk_daophot_opt(args)
    with path.Path(wdir):
        for f in cofes_files:
            # first need to shorten file names such
            # that daophot won't choke on them.
            h,t = os.path.split(f)
            prefix = t[5:20] + t[22:26]
            os.rename(t, prefix + ".fits")
            # execute daophot
            daophot.daophot_find(prefix, args.daophot_sigma)
            # filter ouput
            daophot.filter_daophot_out(prefix + ".coo", prefix + ".lst", args.daophot_xmin,args.daophot_xmax,args.daophot_ymin,args.daophot_ymix)


def daophot_phot_and_allstar(args, cofes_files):
    # run initial daophot phot & allstar
    print("Running daophot phot & allstar ...")
    # Copy configuration files for daophot and allstar.
    shutil.copy2(args.daophot_photo_opt, os.path.join(wdir, "photo.opt") )
    shutil.copy2(args.daophot_allstar_opt, os.path.join(wdir, "allstar.opt") )
    shutil.copy2(args.daophot_phot_psf, os.path.join(wdir, "use.psf"))
    with path.Path(wdir):
        for f in cofes_files:
            # first need to shorten file names such
            # that daophot won't choke on them.
            h,t = os.path.split(f)
            prefix = t[5:20] + t[22:26]
            daophot.daophot_phot(prefix)
            daophot.allstar(prefix)

cofes_files = copy_postage_stamps(args)
#create_postage_stamp_matrix(args, cofes_files)
#inital_daophot_find(args, cofes_files)
daophot_phot_and_allstar(args, cofes_files)

#if __name__ == "__main__":
#    sys.exit(main())




