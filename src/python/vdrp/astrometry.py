""" Astrometry routine

Module to add astrometry to HETDEX catalgoues and images
Contains python translation of Karl Gebhardt

.. moduleauthor:: Maximilian Fabricius <mxhf@mpe.mpg.de>
"""
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
from astropy.io import fits

import path
from astropy import table
from astropy.table import Table

from astropy.stats import biweight_location as biwgt_loc
from astropy.io import fits

from pyhetdex.het import fplane

from vdrp.cofes_vis import cofes_4x4_plots
from vdrp import daophot
from vdrp import cltools

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
    defaults["option"]  = 1
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
    defaults["getoff2_radii"] = 11.,5.,3.
    defaults["mkmosaic_angoff"] = 1.8
    defaults["task"] = "all"

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
    parser.add_argument("--daophot_opt",  type=str)
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
    parser.add_argument("--acam_magadd",  type=float) 
    parser.add_argument("--wfs1_magadd",  type=float)
    parser.add_argument("--wfs2_magadd",  type=float)
    parser.add_argument("--add_radec_angoff",  type=float)
    parser.add_argument('--getoff2_radii', type=str)
    parser.add_argument("--mkmosaic_angoff",  type=float)
    parser.add_argument("-t", "--task",  type=str)

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

    # shoudl in principle be able to do this with accumulate???
    args.getoff2_radii = [float(t) for t in args.getoff2_radii.split(",")]
    #print(args.accumulate(args.getoff2_radii))

    return args


def createDir(directory):
    """ Creates a directory.
    Does not raise an excpetion if the directory already exists.

    Args:
        directory (string): Name for directory to create.
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        logging.error('Creating directory. ' +  directory)


def cp_post_stamps(args, wdir):
    """ Copy CoFeS (collapsed IFU images). 

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
    """
    ## find the IFU postage stamp fits files and copy them over
    logging.info("Copy CoFeS* files to {}".format(wdir))
    pattern = os.path.join( args.reduction_dir, "{}/virus/virus0000{}/*/*/CoFeS*".format( args.night, args.shotid ) )
    cofes_files = glob.glob(pattern)
    already_warned = False
    for f in cofes_files:
        h,t = os.path.split(f)
        if os.path.exists(os.path.join(wdir,t)):
            if not already_warned:
                logging.warning("{} already exists in {}, skipping, won't warn about other files....".format(t,wdir))
                already_warned = True
            continue
        shutil.copy2(f, wdir)


def mk_post_stamp_matrix(args,  wdir, cofes_files):
    """ Create the IFU postage stamp matrix image.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
        cofes_files (list): List of CoFeS file names (collapsed IFU images).
    """
    # create the IFU postage stamp matrix image
    logging.info("Creating the IFU postage stamp matrix image...")
    prefix = os.path.split(cofes_files[0])[-1][:20]
    outfile_name = prefix + ".png"
    with path.Path(wdir):
        cofes_4x4_plots(prefix = prefix, outfile_name = outfile_name, vmin = args.cofes_vis_vmin, vmax = args.cofes_vis_vmax, logging=logging)


def rename_cofes(args,  wdir, cofes_files):
    """ Rename CoFeS files to PREFIX_IFU.fits naming scheme.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (string): Work directory.
        cofes_files (list): List of CoFeS file names (collapsed IFU images).
    """
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
            s  = " '{:30s}'     0.000     0.000   1.00000   0.00000   0.00000   1.00000     0.000    0.0000\n".format(exposures[0] + "tot.als")
            s += " '{:30s}'     1.270    -0.730   1.00000   0.00000   0.00000   1.00000     0.000    0.0000\n".format(exposures[1] + "tot.als")
            s += " '{:30s}'     1.270     0.730   1.00000   0.00000   0.00000   1.00000     0.000    0.0000\n".format(exposures[2] + "tot.als")
            fout.write(s)


def rmaster(args,wdir):
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


def getNorm(all_raw, mag_max ):
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

        logging.info("Rerunning shuffle for RA = {}, Dec = {} and track = {} ...".format(RA0, DEC0, track))
        cmd  = "do_shuffle -v --acam_magadd {:.2f} --wfs1_magadd {:.2f} --wfs2_magadd {:.2f}".format(args.acam_magadd, args.wfs1_magadd, args.wfs2_magadd)
        cmd += " {:.6f} {:.6f} {:.1f} {:d} {:d} {:.1f} {:.1f}".format(RA0, DEC0, radius, track, ifuslot, x_offset, y_offset )
        logging.info("redo_shuffle: Calling shuffle with {}.".format(cmd))
        subprocess.call(cmd, shell=True)

def get_ra_dec_orig(args,wdir):
    """
    Reads first of the many multi* file'd headers to get the RA, DEC, PA guess from the telescope.

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
        with open("radec.orig",'w') as f:
            s = "{} {} {}\n".format(ra0, dec0, pa0)
            f.write(s)

def add_ra_dec(args, wdir, exp_prefixes, ifugrid_file, ra, dec, pa, radec_outfile='tmp.csv'):
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
    # read IFU grid definition file (needs to be replaced by fplane.txt)
    ifugird = Table.read(ifugrid_file, format='ascii')

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


def compute_offset(args, wdir, prefixes, shout_ifustars = 'shout.ifustars'):
    """
    Requires, fplane.txt, radec.orig.
    Creates primarely EXPOSURE_tmp.csv but also radec2.dat.

    Compute offset in RA DEC  by matching detected stars in IFUs
    against the shuffle profived RA DEC coordinates.
    Analogous to rastrom3.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
        shout_ifustars (str): Shuffle output catalog of IFU stars.
    """
    shout_ifustars = 'shout.ifustars'

    ifugrid_file =  os.path.abspath( args.mktot_ifu_grid )

    with path.Path(wdir):
        radii = args.getoff2_radii
        # collect the prefixes that belong to the first exposure
        # for now only do first exposure, later can do them all
        exposures = np.sort( np.unique([p[:15] for p in prefixes]) )
        exp = exposures[0] # select first exposue
        exp_prefixes = []
        # collect all als files for this exposure
        for prefix in prefixes:
            if not prefix.startswith(exp):
                continue
            exp_prefixes.append(prefix)


        # Convert radec.orig to radec.dat, convert RA to degress and add angular offset
        # mF: Not sure if we will need radec.dat later, creating it for now.
        with open("radec.orig") as fradec:
            ll = fradec.readlines()
        tt = ll[0].split()
        ra,dec,pa = float(tt[0]), float(tt[1]), float(tt[2])
        with open("radec.dat",'w') as fradec:
            s = "{} {} {}".format(ra*15.,dec,pa + args.add_radec_angoff)
            fradec.write(s)


        # Now compute offsets iteretively with increasingly smaller matching radii.
        # Matching radii are defined in config file.
        ra_offset, dec_offset = 0., 0.
        for i, radius in enumerate(radii):
            logging.info("Start getoff2 iteration {}, matching radius = {}\"".format(i+1, radius))
            radec_outfile='tmp.csv'
            logging.info("Adding RA & Dec to detections, applying offsets ra_offset,dec_offset,pa_offset = {},{},{}".format( ra_offset, dec_offset, args.add_radec_angoff) )
            # call add_ra_dec, add offsets first.
            new_ra, new_dec, new_pa = ra * 15. + ra_offset, dec + dec_offset, pa + args.add_radec_angoff
            add_ra_dec(args, wdir, exp_prefixes, ifugrid_file, ra=new_ra, dec=new_dec, pa=new_pa, radec_outfile=radec_outfile)
            logging.info("Computing offsets ...")
            dra_offset, ddec_offset = cltools.getoff2(radec_outfile, shout_ifustars, radius, ra_offset=0., dec_offset=0., logging=logging)
            ra_offset, dec_offset =  ra_offset+dra_offset, dec_offset+ddec_offset
            logging.info("End getoff2 iteration {}: Offset get adjusted by {:.6f}, {:.6f} to {:.6f}, {:.6f}".format(i+1, dra_offset, ddec_offset, ra_offset, dec_offset))
            logging.info("")
            logging.info("")

        # read ra,dec, pa from radec.dat
        with open("radec.dat",'r') as f:
            l = f.readline()
            tt = l.split()
            ra,dec,pa = float(tt[0]), float(tt[1]), float(tt[2])

        # write results to radec2.dat
        with open("radec2.dat",'w') as f:
            s = "{} {} {}\n".format(ra+ra_offset,dec+dec_offset,pa )
            f.write(s)


def compute_offset_old(args,wdir,radec_outfiles):
    """
    Requires, fplane.txt, radec.orig.
    Creates primarely EXPOSURE_tmp.csv but also radec2.dat.

    Compute offset in RA DEC  by matching detected stars in IFUs
    against the shuffle profived RA DEC coordinates.
    Analogous to rastrom3.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
        radec_outfiles (list): List file names that contain the ad_ra_dec outputs.
    """
    with path.Path(wdir):
        radii = args.getoff2_radii
        # preparing to do all three exposures here, for now will only do first
        for k in radec_outfiles:
            logging.info("Computing offsets for {}".format(radec_outfiles[k]))
            ra_offset, dec_offset = 0., 0.
            for i, radius in enumerate(radii):
                #fnout = "radec_iter{}.dat" .format(i+1)
                logging.info("Start getoff2 iteration {}".format(i+1))
                ra_offset, dec_offset = cltools.getoff2(radec_outfiles[k], "shout.ifustars", radius, ra_offset, dec_offset, logging=logging)
                logging.info("End getoff2 iteration {}  ra_offset, dec_offset = {}, {}".format(i+1, ra_offset, dec_offset))
                logging.info("")
                logging.info("")

            # read ra,dec, pa from radec.dat
            with open("radec.dat",'r') as f:
                l = f.readline()
                tt = l.split()
                ra,dec,pa = float(tt[0]), float(tt[1]), float(tt[2])

            # write results to radec2.dat
            with open("radec2.dat",'w') as f:
                s = "{} {} {}\n".format(ra+ra_offset,dec+dec_offset,pa )
                f.write(s)

def add_ifu_xy(args, wdir):
    """ Adds IFU x y information to stars used for matching,
    and save to xy.dat.
    Requires: getoff.out, radec2.dat
    Analogous to rastrom3.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        wdir (str): Work directory.
    """
    logging.info("Creating xy.dat...")
    with path.Path(wdir):
        # read ra dec postions of reference stars and detections
        # from getoff.out
        # Produce new tables that add_ifu_xy understands.
        fngetoff_out = "getoff.out"
        t = Table.read("getoff.out", format="ascii.fast_no_header")
        t1 = Table([t['col3'], t['col4'], t['col7']], names=["RA","DEC","IFUSLOT"])
        t2 = Table([t['col5'], t['col6'], t['col7']], names=["RA","DEC","IFUSLOT"])
        t1.write("t1.csv", format="ascii.fast_csv", overwrite=True)
        t2.write("t2.csv", format="ascii.fast_csv", overwrite=True)

        # read ra,dec, pa from radec2.dat
        with open("radec2.dat") as f:
            l = f.readline()
            tt = l.split()
            ra,dec,pa = float(tt[0]), float(tt[1]), float(tt[2])

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
            t4.columns[c].name = t4.columns[c].name + "1"

        t = table.hstack([t3,t4])
        t.write('xy.dat', format="ascii.fixed_width", delimiter='', overwrite=True)
        # this would be analogous to Karl's format
        #t.write('xy.dat', format="ascii.fast_no_header"


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
        with open('radec2.dat','r') as f:
            l = f.readline()
        tt = l.split()
        alpha = 360. - float(tt[2]) + 90. + args.mkmosaic_angoff
        ra,dec = float(tt[0]), float(tt[1])

        logging.info("mkmosaic: Calling imrot (can take a minute) ....")
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

        hdu.writeto("{}fp.fits".format(args.night, args.shotid),overwrite=True)


def get_cofes_files(wdir):
   """
   Create list of all CoFeS* files in the current directory.

   Args:
       wdir (str): Work directory.
   """
   ff = []
   with path.Path(wdir):
       ff = glob.glob('CoFeS????????T??????.?_???_sci.fits')
   return ff


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


def main():
    """
    Main function.
    """
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


    tasks = args.task.split(",")

    for task in tasks:
        if task in ["cp_post_stamps","all"]:
            # Copy over collapsed IFU cubes, aka IFU postage stamps.
            cp_post_stamps(args, wdir)

        elif task in ["mk_post_stamp_matrix","all"]:
            # Creat IFU postage stamp matrix image.
            mk_post_stamp_matrix(args, wdir, get_cofes_files(wdir))

        elif task in ["rename_cofes","all"]:
            # Rename IFU postage stamps as daophot can't handle long file names.
            rename_cofes(args,  wdir, get_cofes_files(wdir))

        elif task in ["daophot_find","all"]:
            # Run initial object detection in postage stamps.
            daophot_find(args, wdir, get_prefixes(wdir))

        elif task in ["daophot_phot_and_allstar","all"]:
            # Run photometry 
            daophot_phot_and_allstar(args, wdir, get_prefixes(wdir))

        elif task in ["mktot","all"]:
            # Combine detections accross all IFUs.
            mktot(args, wdir, get_prefixes(wdir))

        elif task in ["rmaster","all"]:
            # Run daophot master to ???
            rmaster(args, wdir)

        elif task in ["flux_norm","all"]:
            # Compute relative flux normalisation.
            flux_norm(args, wdir)

        elif task in ["redo_shuffle","all"]:
            # Rerun shuffle to get IFU stars
            redo_shuffle(args, wdir)

        elif task in ["get_ra_dec_orig","all"]:
            # Retrieve original RA DEC from one of the multi files.
            # store in radec.orig
            get_ra_dec_orig(args, wdir)

        elif task in ["compute_offset","all"]:
            # Compute offsets by matching 
            # detected stars to sdss stars from shuffle.
            # This also calls add_ra_dec to add RA DEC information to detections.
            compute_offset(args,wdir,get_prefixes(wdir))

        elif task in ["add_ifu_xy","all"]:
            add_ifu_xy(args, wdir)

        elif task in ["mkmosaic","all"]:
            # build mosaic for focal plane
            mkmosaic(args, wdir, get_prefixes(wdir))

        else:
            logging.error("Task {} unknown.".format(task))

    logging.info("Done.")


if __name__ == "__main__":
    sys.exit(main())
