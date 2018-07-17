import numpy as np
from numpy import loadtxt
import argparse
import os
import glob
import shutil
import sys
import ConfigParser
import logging

import path
from astropy.table import Table
from astropy.stats import biweight_location as biwgt_loc

from vdrp.cofes_vis import cofes_4x4_plots
from vdrp import daophot

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
    defaults["option"]  = "default"
    defaults["logfile"] = "astrometry.log"
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
    parser.add_argument("--mktot_ifu_grid",  type=str)
    parser.add_argument("--mktot_magmin",  type=float)
    parser.add_argument("--mktot_magmax",  type=float)
    parser.add_argument("--mktot_xmin",  type=float)
    parser.add_argument("--mktot_xmax",  type=float)
    parser.add_argument("--mktot_ymin",  type=float)
    parser.add_argument("--mktot_ymax",  type=float)
    parser.add_argument("--logfile",  type=str)
    parser.add_argument("--fluxnorm_mag_max",  type=float)
    parser.add_argument("--fplane_txt",  type=str)
    parser.add_argument("--shuffle_cfg",  type=str)

    # positional arguments
    parser.add_argument('night', metavar='night', type=str,
            help='Night of observation (e.g. 20180611).')
    parser.add_argument('shotid', metavar='shotid', type=str,
            help='Shot ID (e.g. 017).')
    parser.add_argument('ra', metavar='ra', type=float,
            help='RA of the target in decimal hours.')
    parser.add_argument('dec', metavar='dec', type=float,
            help='Dec of the target in decimal hours degree.')
    parser.add_argument('track', metavar='track', type=int, , choices=[0, 1],
            help='Type of track: 0: East 1: West')


    args = parser.parse_args(remaining_argv)
 
    return args


def createDir(directory):
    global logging
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        logging.error('Creating directory. ' +  directory)


def copy_postage_stamps(args, wdir):
    global logging
    ## find the IFU postage stamp fits files and copy them over
    logging.info("Copy CoFeS* files to {}".format(wdir))
    pattern = os.path.join( args.reduction_dir, "{}/virus/virus0000{}/*/*/CoFeS*".format( args.night, args.shotid ) )
    cofes_files = glob.glob(pattern)
    already_warned = False
    for f in cofes_files:
        h,t = os.path.split(f)
        if os.path.exists(os.path.join(wdir,t)):
            if not already_warned:
                print("{} already exists in {}, skipping, won't warn about other files....".format(t,wdir))
                already_warned = True
            continue
        shutil.copy2(f, wdir)
    return cofes_files

def create_postage_stamp_matrix(args,  wdir, cofes_files):
    global logging
    # create the IFU postage stamp matrix image
    logging.info("Creating the IFU postage stamp matrix image...")
    prefix = os.path.split(cofes_files[0])[-1][:20]
    outfile_name = prefix + ".png"
    with path.Path(wdir):
        cofes_4x4_plots(prefix = prefix, outfile_name = outfile_name, vmin = args.cofes_vis_vmin, vmax = args.cofes_vis_vmax, logging=logging)


def rename_cofes(args,  wdir, cofes_files):
    global logging
    # run initial daophot find
    logging.info("Renaming CoFeS* files ...")
    prefixes = []
    with path.Path(wdir):
        for f in cofes_files:
            # first need to shorten file names such
            # that daophot won't choke on them.
            h,t = os.path.split(f)
            prefix = t[5:20] + t[22:26]
            prefixes.append(prefix)
            os.rename(t, prefix + ".fits")
    return prefixes


def inital_daophot_find(args,  wdir, prefixes):
    global logging
    # run initial daophot find
    logging.info("Running initial daophot find...")
    # Create configuration file for daophot.
    prefixes = []
    with path.Path(wdir):
        daophot.mk_daophot_opt(args)
        for prefix in prefixes:
            # execute daophot
            daophot.daophot_find(prefix, args.daophot_sigma)
            # filter ouput
            daophot.filter_daophot_out(prefix + ".coo", prefix + ".lst", args.daophot_xmin,args.daophot_xmax,args.daophot_ymin,args.daophot_ymix)
    return prefixes


def daophot_phot_and_allstar(args, wdir, prefixes):
    """
    Runs daophot photo and allstar on all IFU postage stamps.
    Produces *.ap and *.als files.
    Analogous to run4a.
    """
    global logging
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
            h,t = os.path.split(f)
            prefix = t[5:20] + t[22:26]
            daophot.daophot_phot(prefix)
            daophot.allstar(prefix)


def mktot(args, wdir, prefixes):
    """
    Read all *.als files. Put detections on a grid
    corresponding to the IFU position in the focal plane as defined in
    config/ifu_grid.txt (should later become fplane.txt.
    Then produce all.mch.
    (Analogous to run6 and run6b)
    """
    global logging

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
                    yoff = ifugird['X'][jj][0]

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
                        s = "{:03d} {:8.3f} {:8.3f} {:8.3f}\n".format( d['ID'], d['X']+xoff, d['Y']+yoff, d['MAG'] )
                        fout.write(s)

                logging.info("{} stars in {}.".format(count, fnout))
        # produce all.mch like run6b
        with open("all.mch", 'w') as fout:
            s  = " '{:30s}'     0.000     0.000   1.00000   0.00000   0.00000   1.00000     0.000    0.0000\n".format(exposures[0] + "tot.als")
            s += " '{:30s}'     1.270    -0.730   1.00000   0.00000   0.00000   1.00000     0.000    0.0000\n".format(exposures[1] + "tot.als")
            s += " '{:30s}'     1.270     0.730   1.00000   0.00000   0.00000   1.00000     0.000    0.0000\n".format(exposures[2] + "tot.als")
            fout.write(s)


def rmaster(args,wdir):
    """
    Executed daomaster.
    Analogous to run8b.
    """
    global logging
    logging.info("Running daomaster.")
    with path.Path(wdir):
        daophot.daomaster()


def getNorm(all_raw, mag_max ):
    """
    Comutes the actual normalisation for fluxNorm.
    Analogous to run9.
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


def fluxNorm(args, wdir, infile='all.raw', outfile='norm.dat'):
    """
    Reads all.raw and compute relative flux normalisation
    for the three exposures.
    Analogous to run9.
    """
    global logging
    logging.info("Computing flux normalisation between exposures 1,2 and 3.")
    mag_max = args.fluxnorm_mag_max
    with path.Path(wdir):
        all_raw = loadtxt(infile, skiprows=3)
        n1,n2,n3 = getNorm(all_raw, mag_max )
        logging.info("Flux normalisation is {:10.8f} {:10.8f} {:10.8f}".format(n1,n2,n3) )
        with open(outfile, 'w') as f:
            s = "{:10.8f} {:10.8f} {:10.8f}".format(n1,n2,n3)
            f.write(s)


# Parse config file and command line paramters
# command line parameters overwrite config file.
args = parseArgs()

# Set up logging to file - see previous section for more details
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    handlers=[
                        logging.FileHandler(args.logfile),
                        logging.StreamHandler()
                    ],
                    filemode='w')

# Create work directory for given night and shot
cwd = os.getcwd()
wdir = os.path.join(cwd, "{}v{}".format(args.night, args.shotid))
createDir(wdir)

# Copy over collapsed IFU cubes, aka IFU postage stamps.
cofes_files = copy_postage_stamps(args, wdir)

# Creat IFU postage stamp matrix image.
#create_postage_stamp_matrix(args, wdir, cofes_files)

# Rename IFU postage stamps as daophot can't handle long file names.
prefixes = rename_cofes(args,  wdir, cofes_files)

# Run initial object detection in postage stamps.
#inital_daophot_find(args, prefixes)

# Run photometry 
#daophot_phot_and_allstar(args, prefixes)

# Combine detections accross all IFUs.
#mktot(args, wdir, prefixes)

# Run daophot master to ???
#rmaster(args, wdir)

# Compute relative flux normalisation.
#fluxNorm(args, wdir)

import subprocess

def redo_shuffle(args):
    logging.info("Rerunning shuffle for RA = {}, Dec = {} and track = {} ...".format(RA0, DEC0, track))
    shutil.copy2(args.shuffle_cfg, wdir)
    shutil.copy2(args.fplane_txt, wdir)
    with Path(wdir):
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

    cmd  = "do_shuffle -v --acam_magadd {:d} --wfs1_magadd {:d --wfs2_magadd {:d}".format(acam_magadd, wfs1_magadd, wfs2_magadd)
    cmd += "{:.6f} {:.6f} {:.1f} {:d} {:d} {:.1f} {:.1f}".format(RA0, DEC0, radius, track, ifuslot, x_offset, y_offset )

    subprocess.call(cmd, shell=True)


from astropy.io import fits
def get_ra_dec_orig(args):
    logging.info("")
    pattern = os.path.join( args.reduction_dir, "/$1/virus/virus0000$2/*/*/multi_???_*LL*fits".format( args.night, args.shotid ) )
    multi_files = glob.glob(pattern)
    h = fits.getheader(multi_files[0])
    ra0  = h["TRAJCRA"]
    dec0 = h["TRAJCDEC"]
    pa0  = h["PARANGLE"]
    with open("radec.orig") as f:
        s = "{} {} {}".format(ra0, dec0, pa0)
        f.write(s)


#20180611v017"
#grep -v "#" shout.ifustars > NIGHT.ifu

#if __name__ == "__main__":
#    sys.exit(main())




