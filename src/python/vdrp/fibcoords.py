#!/usr/bin/env python
""" Astrometry routine

Module to add astrometry to HETDEX catalgoues and images
Contains python translation of Karl Gebhardt

.. moduleauthor:: Maximilian Fabricius <mxhf@mpe.mpg.de>
"""
from __future__ import print_function

import numpy as np
import argparse
import os
import glob
import shutil
import sys
import ConfigParser
import logging
from collections import OrderedDict
import path
import ast

from astropy.io import ascii
from astropy.table import Table
from astropy.table import Column
from astropy.table import vstack

from pyhetdex.het.fplane import FPlane
from pyhetdex.coordinates.tangent_projection import TangentPlane
import pyhetdex.tools.read_catalogues as rc

from vdrp.utils import createDir
from vdrp.utils import read_radec
from vdrp.utils import rm
from vdrp.fplane_client import retrieve_fplane

def parseArgs():
    """ Parses configuration file and command line arguments.
    Command line arguments overwrite configuration file settiongs which
    in turn overwrite default values.

    Args:
        args (argparse.Namespace): Return the populated namespace.
    """

    # Do argv default this way, as doing it in the functional
    # declaration sets it at compile time.
    argv = None
    if argv is None:
        argv = sys.argv

    # Parse any conf_file specification
    # We make this parser with add_help=False so that
    # it doesn't parse -h and print help.
    conf_parser = argparse.ArgumentParser(
            description=__doc__,  # printed with -h/--help
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
    defaults["ixy_dir"] = "vdrp/config/"
    defaults["shifts_dir"] = "shifts/"
    defaults["fplane_txt"] = "vdrp/config/fplane.txt"
    defaults["radec2_dat"] = ""
    defaults["dither_offsets"] = "[(0.,0.),(1.270,-0.730),(1.270,0.730)]"

    if args.conf_file:
        config = ConfigParser.SafeConfigParser()
        config.read([args.conf_file])
        defaults.update(dict(config.items("Astrometry")))

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h

    # Inherit options from config_parser
    parser = argparse.ArgumentParser(parents=[conf_parser])
    parser.set_defaults(**defaults)
    parser.add_argument("--logfile", type=str)
    parser.add_argument("--addin_dir", type=str)
    parser.add_argument("--ixy_dir", type=str)
    parser.add_argument("--shifts_dir", type=str)
    parser.add_argument("--fplane_txt",  type=str,
                        help="filename for fplane file.")
    parser.add_argument("--radec2_dat", type=str,
                        help="Overwrite use of default radec2_final.dat "
                        "(holds final astrometric solution for shot "
                        "RA,Dec,PA) by specific file.")
    parser.add_argument("--dither_offsets", type=str,
                        help="List of x,y tuples that define "
                        "the dither offsets.")

    #  positional arguments
    parser.add_argument('night', metavar='night', type=str,
                        help='Night of observation (e.g. 20180611).')
    parser.add_argument('shotid', metavar='shotid', type=str,
                        help='Shot ID (e.g. 017).')

    args = parser.parse_args(remaining_argv)

    args.dither_offsets = ast.literal_eval(args.dither_offsets)

    return args


def get_exposures(reduction_dir, night, shotid):
    """
    Search reductions directory and find how many exposures there are.

    Args:
        reduction_dir (str): Directory that holds panacea reductions. Expects
            subdirectories like ./NIGHTvSHOT.
        night (str): Night for and observation.
        shot (str): Shot ID for the observation.

    Returns:
        (list): Sorted list with exposures in reduction directory for night
                and shot (from args). E.g. ['exp01', exp02', 'exp03'].
    """
    pattern = os.path.join(reduction_dir,
                           "{}/virus/virus0000{}/exp??".format(night,
                                                               shotid))
    expdirs = glob.glob(pattern)
    exposures = []
    for d in expdirs:
        __, t = os.path.split(d)
        exposures.append(t)

    logging.info("get_exposures: Found {} exposures.".format(len(exposures)))
    return np.sort(exposures)


def get_active_slots(wdir, exposures):
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
            pattern = "{}/virus/multi*".format(exp)
            ff = glob.glob(pattern)
            exp_slots[exp] = []
            for f in ff:
                # could get ifuslot from filname
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


def mk_exp_sub_dirs(wdir, exposures):
    """ Make subdirectory structure.
    Creates ./exp??/virus

    Args:
        wdir (str): Work directory.
        exposures (list): Sorted list with exposures in reduction
                          directory for night and shot
                          (from args). E.g. ['exp01', exp02', 'exp03'].

    """
    logging.info("Creating exp0?/virus structure.")
    with path.Path(wdir):
        for exp in exposures:
            createDir(exp + "/virus")


def mk_coords_sub_dir(wdir):
    """ Make subdirectory structure.
    Creates ./coords

    Args:
        wdir (str): Work directory.
        exposures (list): Sorted list with exposures in
                          reduction directory for night and shot
                (from args). E.g. ['exp01', exp02', 'exp03'].

    """
    logging.info("Creating coords subdirectory.")
    with path.Path(wdir):
        createDir("coords")


def create_elist(wdir, reduction_dir, night, shotid, exposures):
    """ Creates elist file. For each exposure there will be one
    entry for the exposre number and the
    timestamp like

        exp01	20180611T054545.2
        exp02	20180611T055249.6
        exp03	20180611T060006.3

    Args:
        wdir (str): Work directory.
        reduction_dir (str): Directory that holds panacea reductions. Expects
            subdirectories like ./NIGHTvSHOT.
        night (str): Night for and observation.
        shotid (str): Shot ID for the observation.
        exposures (list): Sorted list with exposures in reduction
                          directory for night and shot
                (from args). E.g. ['exp01', exp02', 'exp03'].

    """
    logging.info("Creating elist.")
    with path.Path(wdir):
        with open("elist", "w") as felist:
            for exp in exposures:
                pattern = os.path.join(reduction_dir,
                                       "{}/virus/virus0000{}/{}/virus/CoFeS*"
                                       .format(night, shotid, exp))
                logging.info("    using CoFes*fits files {}..."
                             .format(pattern))
                CoFeS = glob.glob(pattern)
                __, t = os.path.split(CoFeS[0])
                prefix = t[5:22]
                felist.write("{} {}\n".format(exp, prefix))


def link_multifits(wdir, reduction_dir, night, shotid, exposures):
    """ Link Panacea's multifits files into exp??/virus subdiretories.

    Args:
        wdir (str): Work directory.
        reduction_dir (str): Directory that holds panacea reductions. Expects
            subdirectories like ./NIGHTvSHOT.
        night (str): Night for and observation.
        shotid (str): Shot ID for the observation.
        exposures (list): Sorted list with exposures in reduction
                          directory for night and shotid
                (from args). E.g. ['exp01', exp02', 'exp03'].

    """
    logging.info("Creating links to multi*fits files.")
    with path.Path(wdir):
        for exp in exposures:
            pattern = os.path.join(reduction_dir,
                                   "{}/virus/virus0000{}/{}/virus/multi*"
                                   .format(night, shotid, exp))
            logging.info("    Creating links to multi*fits files {}..."
                         .format(pattern))
            multifits = glob.glob(pattern)
            logging.info("    Linking {} files ...".format(len(multifits)))
            for mf in multifits:
                __, t = os.path.split(mf)
                target = os.path.join(wdir, "{}/virus/{}".format(exp, t))
                rm([target])
                os.symlink(mf, os.path.join(wdir, target))


def cp_astrometry(wdir, shifts_dir, night, shotid, radec2_dat):
    """ Copies astrometry information from
    shifts directory.

    Args:
        wdir (str): Work directory.
        shifts_dir (str): Directory where the astrometry was calculated,
                          usually "shifts".
        night (str): Night for and observation.
        shotid (str): Shot ID for the observation.
        radec2_dat (str): Default '', if not empty will overwrite the
                          use of the default radec2_final.dat
                          from the shifts directory.

    Notes:
        If radec2_dat is used to specify different source files
        it will still be named radec2_final.dat in the target directory.
        This is ugly, but those parameters mostly serve for debugging.
    """
    logging.info("Copy over radec2_final.dat from {}.".format(shifts_dir))
    # ff = []

    if radec2_dat == "":
        radec2_dat = "{}/{}v{}/radec2_final.dat".format(shifts_dir, night,
                                                        shotid)
    logging.info("Using {} for RA, Dec and PA.".format(radec2_dat))
    shutil.copy2(radec2_dat, os.path.join(wdir, "coords", "radec2_final.dat"))


def cp_addin_files(wdir, addin_dir, subdir="coords"):
    """ Copies `addin` files. These are
    essentially the IFUcen files in a different format.

    Args:
        addin_dir (str): Directory where the *.addin files are stored.
        wdir (str): Work directory.
    """
    logging.info("Copy over *.addin from {}.".format(addin_dir))
    pattern = addin_dir + "/*.addin"
    ff = glob.glob(pattern)
    logging.info("    Found {} files.".format(len(ff)))
    for f in ff:
        shutil.copy2(f, os.path.join(wdir, subdir))


def cp_ixy_files(wdir, ixy_dir, subdir="coords"):
    """ Copies `ixy` files. These are
    essentially the IFUcen files in a different format.

    Args:
        ixy_dir_dir (str): Directory where the *.ixy files are stored.
        wdir (str): Work directory.
    """
    logging.info("Copy over *.ixy from {}.".format(ixy_dir))
    pattern = ixy_dir + "/*.ixy"
    ff = glob.glob(pattern)
    logging.info("    Found {} files.".format(len(ff)))
    for f in ff:
        shutil.copy2(f, os.path.join(wdir, subdir))


def get_fiber_coords(wdir, active_slots, dither_offsets, subdir="coords"):
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
        This creates a list of files iIFUSLOT_DITHERNUM.csv
        that store the on-sky fiber coordinates.

    Args:
        wdir (str): Work directory.

    """

    logging.info("get_fiber_coords: Computing on-sky fiber coordinates.")
    with path.Path(os.path.join(wdir, subdir)):
        ra0, dec0, pa0 = read_radec("radec2_final.dat")

        # Find which IFU slots to operate of based on the
        # existing set og *.addin files.
        ifuslots = []
        addin_files = []
        for slot in active_slots:
            fn = "{}.addin".format(slot)
            if os.path.exists(fn):
                ifuslots.append(slot)
                addin_files.append(fn)
            else:
                logging.warning("get_fiber_coords: Found no addin file "
                                "for slot {}. This slot delivers data however."
                                .format(slot))

        # Carry out required changes to astrometry
        rot = 360.0 - (pa0 + 90.)

        # Set up astrometry from user supplied options
        tp = TangentPlane(ra0, dec0, rot)

        fplane = FPlane("fplane.txt")
        for offset_index, (dx, dy) in enumerate(dither_offsets):
            logging.info("get_fiber_coords:    offset_index {} dx "
                         "= {:.3f}, dy = {:.3f}."
                         .format(offset_index + 1, dx, dy))
            #print("ifuslots: ", ifuslots)
            #print("addin_files: ", addin_files)
            for ifuslot, addin_file in zip(ifuslots, addin_files):
                # identify ifu
                if not ifuslot in fplane.ifuslots:
                    logging.warning("IFU {} not in fplane file.".format(ifuslot))
                    continue
                ifu = fplane.by_ifuslot(ifuslot)
                # read fiber positions
                x, y, table = rc.read_line_detect(addin_file)
                # skip empty tables
                if len(x) < 1:
                    continue
                # remember to flip x,y
                xfp = x + ifu.y + dx
                yfp = y + ifu.x + dy
                # project to sky
                # print("ifuslot, addin_file, ifu.x, dx, ifu.y , dy, ra0, "
                #       "dec0, pa0", ifuslot, addin_file, ifu.x, dx, ifu.y ,
                #       dy, ra0, dec0, pa0)

                ra, dec = tp.xy2raDec(xfp, yfp)
                # save results
                table['ra'] = ra
                table['dec'] = dec
                table['ifuslot'] = ifuslot
                table['xfplane'] = xfp
                table['yfplane'] = yfp
                outfilename = "i{}_{}.csv".format(ifuslot, offset_index + 1)
                logging.info("Writing {}.".format(outfilename))
                table.write(outfilename, comment='#', format='ascii.csv',
                            overwrite=True)

def config_loggerNew(args, target_dir):
    """ Setup logging to file and screen.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
        target_dir (str): Directory where the logfile shell be saved.
    """

    fmt = '%(asctime)s %(levelname)-8s %(funcName)15s(): %(message)s'
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
        format=fmt,
        datefmt='%m-%d %H:%M',
        filename=os.path.join(target_dir, args.logfile),
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

def config_logger(args, target_dir="20180601v007"):
    """ Setup logging to file and screen.

    Args:
        args (argparse.Namespace): Parsed configuration parameters.
    """

    fmt = '%(asctime)s %(levelname)-8s %(funcName)15s(): %(message)s'
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(target_dir, args.logfile),
                        filemode='a')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter(fmt)
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


def read_elist(filename):
    """
    Reads exposure lsit file.

    Returns:
        (OrderedDict): Dictionary that contains exposure identifier
                       and timestamp like
            {
            "exp01"	: "20180611T054545.2",
            "exp02"	: "20180611T055249.6",
            "exp03"	: "20180611T060006.3"
            }
    """
    with open(filename, 'r') as f:
        ll = f.readlines()
    e = OrderedDict()
    for l in ll:
        tt = l.split()
        e[tt[0]] = tt[1]
    return e


def mk_dithall(wdir, active_slots, subdir="coords", fnelist = "../elist"):
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

        column_names = "ra", "dec", "ifuslot", "XS", "YS", "dxfplane", \
            "yfplane", "multifits", "timestamp", "exposure"
        # read list of exposures
        elist = read_elist(fnelist)
        all_tdith = []
        for i, exp in enumerate(elist):
            logging.info("get_fiber_coords: Exposure {} ...".format(exp))
            exp_tdith = []
            for ifuslot in active_slots:
                # read the ixy files, those contain the mapping of
                # x/y (IFU space) to fiber number on the detector
                ixy_filename = "{}.ixy".format(ifuslot)
                if not os.path.exists(ixy_filename):
                    logging.warning("mk_dithall: Found no *.ixy file for "
                                    "IFU slot {} ({})."
                                    .format(ifuslot, ixy_filename))
                    continue
                # some evil person put tabs in just some of the files ....
                with open(ixy_filename) as f:
                    s = f.read()
                s = s.replace("\t", " ")
                ixy = ascii.read(s, format='no_header')
                csv_filename = "i{}_{}.csv".format(ifuslot, i+1)
                if not os.path.exists(csv_filename):
                    logging.warning("mk_dithall: Found no *.csv file for "
                                    "IFU slot {} ({})."
                                    .format(ifuslot, csv_filename))
                    continue
                csv = ascii.read(csv_filename)

                # pointer to the multi extention fits and fiber
                # nubmer like: multi_301_015_038_RU_085.ixy
                cmulti_name = ixy["col3"]
                cifu = Column(["ifu{}".format(ifuslot)] * len(ixy))
                # cc = csv["ra"], csv["dec"], cifu, csv["ifuslot"], csv["XS"],\
                #     csv["YS"], csv["xfplane"], csv["yfplane"], cmulti_name
                # tdithx = Table(data=cc)
                # tdithx.write("dith{}.all".format(i+1), overwrite=True,
                #              format='ascii.fixed_width')
                cprefix = Column([elist[exp]] * len(ixy))
                cexp = Column([exp] * len(ixy))
                cc = csv["ra"], csv["dec"], cifu, csv["XS"], csv["YS"], \
                    csv["xfplane"], csv["yfplane"], cmulti_name, cprefix, cexp
                tdith = Table(cc, names=column_names)
                all_tdith.append(tdith)
                exp_tdith.append(tdith)
            vstack(exp_tdith).write("dith_{}.all".format(exp), overwrite=True,
                                    format='ascii.fixed_width', delimiter="")

        tall_tdithx = vstack(all_tdith)
        tall_tdithx.write("dithall.use", overwrite=True,
                          format='ascii.fixed_width', delimiter="")


def main():
    """
    Main function.
    """
    # Parse config file and command line paramters
    # command line parameters overwrite config file.
    args = parseArgs()


    # Create results directory for given night and shot
    cwd = os.getcwd()
    wdir = os.path.join(cwd, "{}v{}".format(args.night, args.shotid))
    createDir(wdir)
    # Set up logging
    config_logger(args, wdir)
    logging.info("Start.")


    exposures = get_exposures(args.reduction_dir, args.night, args.shotid)

    # create subdirectories for each exposure and ./coords
    mk_exp_sub_dirs(wdir, exposures)
    mk_coords_sub_dir(wdir)

    # create symlinks to multi*fits fitsfiles
    link_multifits(wdir, args.reduction_dir, args.night,
                   args.shotid, exposures)

    # copy astrometry solution (all.mch and radec2.dat) from shifts directory
    cp_astrometry(wdir, args.shifts_dir, args.night, args.shotid,
                  args.radec2_dat)

    # Copy `addin` files. These are essentially the IFUcen files
    # in a different format.
    cp_addin_files(wdir, args.addin_dir)

    # Copy `ixy` files. These conain the IFU x/y to fiber number mapping.
    cp_ixy_files(wdir, args.ixy_dir)

    # Copy fplane file.
    retrieve_fplane(args.night, args.fplane_txt, os.path.join(wdir, "coords") )

    # find which slots delivered data for all exposures
    # (infer from existance of corresponding multifits files).
    active_slots = get_active_slots(wdir, exposures)

    # This is where the main work happens.
    # Essentially calls add_ra_dec for all IFU slots and all dithers.
    # Actually it uses the direct python interface to tangent_plane
    # rhater than calling add_ra_dec.
    get_fiber_coords(wdir, active_slots, args.dither_offsets)

    # Create exposure list
    create_elist(wdir, args.reduction_dir, args.night, args.shotid, exposures)

    # Create final dithall.use file for downstream functions.
    mk_dithall(wdir, active_slots)

    logging.info("Done.")

    sys.exit(0)


if __name__ == "__main__":
    sys.exit(main())
