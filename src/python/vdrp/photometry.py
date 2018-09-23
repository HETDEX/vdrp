#!/usr/bin/env python
""" Photometry routine

Contains python translation of Karl Gebhardt

.. moduleauthor:: Maximilian Fabricius <mxhf@mpe.mpg.de>
"""

from __future__ import print_function
# import matplotlib

# from matplotlib import pyplot as plt

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
# from astropy.io import ascii
# import tempfile
import numpy as np
from collections import OrderedDict
import pickle
# import ast

# import scipy
# from scipy.interpolate import UnivariateSpline

from distutils import dir_util

# from astropy import table
# from astropy.table import Table

# from astropy.stats import biweight_location as biwgt_loc
# from astropy.table import vstack

# from pyhetdex.het import fplane
# from pyhetdex.coordinates.tangent_projection import TangentPlane
# import pyhetdex.tools.read_catalogues as rc
# from pyhetdex import coordinates
# from pyhetdex.coordinates import astrometry as phastrom

# from vdrp.cofes_vis import cofes_4x4_plots
# from vdrp import daophot
# from vdrp import cltools
# from vdrp import utils
import utils
# from vdrp.daophot import DAOPHOT_ALS
# from vdrp.utils import read_radec, write_radec

# matplotlib.use("agg")


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


class Spectrum():
    """
    This class encapsulates the content of a tmp*.dat spectrum file
    """
    def __init__(self):
        self.wl = None
        self.cnts = None
        self.flux = None
        self.amp_norm = None
        self.tp_norm = None
        self.ftf_norm = None
        self.err_cts = None
        self.err_cts_local = None
        self.err_max_flux = None

    def read(self, fname):
        indata = np.loadtxt(fname).transpose()

        self.wl = indata[0]
        self.cnts = indata[1]
        self.flux = indata[2]
        self.amp_norm = indata[3]
        self.tp_norm = indata[4]
        self.ftf_norm = indata[5]
        self.err_cts = indata[6]
        self.err_cts_local = indata[7]
        self.err_max_flux = indata[8]


class ShuffleStar():

    def __init__(self, starid=-1, shotid=-1, shuffleid=-1, ra=-1.0, dec=-1.0,
                 u=99., g=99., r=99., i=99., z=99.):
        self.starid = -1
        self.shotid = shotid
        self.shuffleid = shuffleid
        self.ra = ra
        self.dec = dec
        self.mag_u = u
        self.mag_g = g
        self.mag_r = r
        self.mag_i = i
        self.mag_z = z


class StarObservation():

    def __init__(self, night=-1, shot=-1, ra=-1, dec=-1, x=-1, y=-1, fname='',
                 shotname='', expname='', offset_ra=-1, offset_dec=1):

        self.night = night
        self.shot = shot
        self.ra = ra
        self.dec = dec
        self.x = x
        self.y = y
        self.full_fname = fname
        self.shotname = shotname
        self.expname = expname
        self.offset_ra = offset_ra
        self.offset_dec = offset_dec
        self.fname = ''
        self.ifuslot = ''

        self.fname, self.ifuslot = self.full_fname.split('.')[0].rsplit('_', 1)


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
    defaults['logfile'] = 'photometry.log'
    defaults["use_tmp"] = "False"
    defaults["remove_tmp"] = "True"
    defaults["shuffle_mag_limit"] = 20.

    defaults["shuffle_ifustars_dir"] = \
        '/work/00115/gebhardt/maverick/sci/panacea/test/shifts/'
    defaults['multifits_dir'] = '/work/03946/hetdex/maverick/red1/reductions/'
    defaults['tp_dir'] = '/work/00115/gebhardt/maverick/detect/tp/'
    defaults['norm_dir'] = '/work/00115/gebhardt/maverick/getampnorm/all/'
    defaults['bin_dir'] = '/home/00115/gebhardt/bin/'

    defaults['extraction_aperture'] = 1.6
    defaults['extraction_wl'] = 4500.
    defaults['extraction_wlrange'] = 1000.
    defaults['radec_file'] = '/work/00115/gebhardt/maverick/getfib/radec.all'
    defaults['ifu_search_radius'] = 4.

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
    parser.add_argument("--shuffle_mag_limit", type=float,
                        help="Magnitude cutoff for selection of stars found by"
                        " shuffle")
    parser.add_argument("--shuffle_ifustars_dir", type=str, help="Directory "
                        "with the *ifustars shuffle output files")
    parser.add_argument("--multifits_dir", type=str, help="Directory "
                        "with the multi extension fits files")
    parser.add_argument("--tp_dir", type=str, help="Directory "
                        "with the throughput files")
    parser.add_argument("--norm_dir", type=str, help="Directory "
                        "with the amplifier normalization files")
    parser.add_argument("--bin_dir", type=str, help="Directory "
                        "with the fortran binary files.")

    parser.add_argument("--extraction_aperture", type=float, help="Aperture "
                        "radius in asec for the extraction")
    parser.add_argument("--extraction_wl", type=float, help="Central "
                        "wavelength for the extraction")
    parser.add_argument("--extraction_wlrange", type=float, help="Wavelength "
                        "range for the extraction")
    parser.add_argument("--radec_file", type=str, help="Filename of file with "
                        "RA DEC PA positions for all shots")
    parser.add_argument("--ifu_search_radius", type=float, help="Radius for "
                        "search for shots near a given shuffle star.")

    # positional arguments
    parser.add_argument('night', metavar='night', type=str,
                        help='Night of observation (e.g. 20180611).')
    parser.add_argument('shotid', metavar='shotid', type=str,
                        help='Shot ID (e.g. 017).')
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


def run_command(cmd, input=None):
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    so, _ = proc.communicate(input=input)
    for l in so.split("\n"):
        if logging is not None:
            logging.info(l)
        else:
            print(l)
    proc.wait()


def call_imextsp(bindir, filename, ifuslot, wl, wlw, tpavg, norm, outfile):
    """
    Equivalent of the rextsp script,
    a wrapper around the imextsp fortran routine.
    """
    input = '{filename:s}\n{ifuslot} {wl} {wlw}\n {tpavg} {norm}'

    try:
        os.remove('out.sp')
    except os.FileNotFoundError:
        pass

    try:
        os.remove('outfile')
    except os.FileNotFoundError:
        pass

    s = input.format(filename=filename, ifuslot=ifuslot, wl=wl, wlw=wlw,
                     tpavg=tpavg, norm=norm)

    run_command(bindir + '/imextsp', s)

    shutil.move('out.sp', outfile)


def call_sumsplines(bindir, nspec):
    """
    Call sumsplines

    Creates a file called splines.out
    """
    with open('list', 'w') as f:
        for i in range(0, nspec):
            f.write('tmp%d.dat\n' % i+101)

    run_command(bindir + '/sumsplines')


def call_fitonevp(bindir, wave, outfile):
    """
    Call fitonevp

    Requires fitghsp.in created by apply_factor_spline
    """
    input = '0 0\n{wave:f}\n/vcps\n'

    run_command('fitonevp', input.format(wave=wave))

    shutil.move('lines.out', outfile)


def get_throughput_file(path, shotname):
    """
    Equivalent of rtp0 script.

    Checks if a night/shot specific throughput file exists.

    If true, return the filename, otherise the filename
    for an average throughput file.
    """
    if os.path.exists(path + '/' + shotname + "sedtp_f.dat"):
        return path + '/' + shotname + "sedtp_f.dat"
    else:
        return path + '/' + "tpavg.dat"


def apply_factor_spline(factor):
    """
    Equivalent of the rawksp[12] scripts
    """
    wave, flx = np.loadtxt('splines.out', usecols=[0, 2])

    with open('fitghsp.in', 'w') as f:
        for w, f in zip(wave, flx):
            f.write('%f %f' % (w, f*1.e17 / factor))


def read_stuctaz(fname):
    """
    Equivalent of the rgetadc script

    Parameters:
    -----------

    fname : string
    Filename to read from

    Read the STRUCTAZ parameter from the fits file ``fname``
    """

    with fits.open(fname, 'r') as hdu:
        return hdu[0].header['STRUCTAZ']


def get_star_spectrum_data(star, args):
    """
    This extracts the data about the different observations of the same star
    on different ifus.

    This is essentially the information stored in the l1 file.
    """

    # First find matching shots
    night, shot = np.loadtxt(args.radec_file, unpack=True, usecols=[0, 1])
    ra_ifu, dec_ifu, x_ifu, y_ifu = np.loadtxt(args.dithall_file, unpack=True,
                                               usecols=[0, 1, 3, 4])
    fname_ifu, shotname_ifu, expname_ifu = np.loadtxt(args.dithall_file,
                                                      unpack=True,
                                                      usecols=[7, 8, 9])

    # If this code should be run in a more general fashion, use the
    # ra / dec
    w = np.where((night == int(args.night)) & (shot == int(args.shotid))
                 & ((np.sqrt((np.cos(star.dec/57.3)*(ra_ifu-star.ra))**2
                             + (dec_ifu-star.dec)**2)*3600.)
                    < args.ifu_search_radius))

    night_shots = []

    starobs = []

    for i in w:

        so = StarObservation((night[i], shot[i], ra_ifu[i], dec_ifu[i],
                              x_ifu[i], y_ifu[i], fname_ifu[i],
                              shotname_ifu[i], expname_ifu[i]))

        # This is written to loffset
        so.offsets_ra = 3600.*(ra_ifu[i]-star.ra)
        so.offsets_dec = 3600.*(dec_ifu[i]-star.dec)

        starobs.append(so)
        night_shots.append('%d %d' % (night[i], shot[i]))

    return starobs, np.unique(night_shots)


def extract_star_spectrum(starobs, args):
    """
    Equivalent of the rextsp0 and parts of the rsp1b scripts
    """

    c = 0

    for s in starobs:
        call_imextsp(args.bin_dir, args.multifits_dir+'/'+s.fname+'.fits',
                     s.ifuslot, args.extraction_wl, args.extraction_wlrange,
                     get_throughput_file(args.tp_path, s.night+'v'+s.shot),
                     args.norm_path+'/'+s.fname+".norm", 'tmp%d.dat' % c+101)

        c += 1


def get_shuffle_stars(shuffledir, nightshot, maglim):

    stars = []

    c = 1
    try:
        indata = np.loadtxt(shuffledir + '/' + nightshot + '/shout.ifustars')
        for d in indata:
            star = ShuffleStar(d[0], d[1], d[2], d[3], d[4], d[5], d[6],
                               d[7], d[8])
            if star.mag_g < maglim:
                star.starid = 20000 + c
                stars.append(star)
                c += 1

        return stars
    except OSError:
        logging.error('Failed to find shuffle stars for night %s, shot %s'
                      % (args.night, args.shotid))


def average_spectrum(spec, wlmin, wlmax):
    """
    Corresponds to ravgspq script. Calculate the average of the
    spectrum in the range [wlmin, wlmax]

    Parameters
    ----------
    spec : Spectrum
        Spectrum class object
    """

    wh = (spec.wl > wlmin) & (spec.wl < wlmax)

    # Calculate the mean of all values within wavelength range
    # where cnts are !=0

    avg = spec.cnts[wh][spec.cnts[wh] != 0.].mean()
    norm = (spec.amp_norm[wh]*spec.tp_norm[wh]).sum()

    return avg, norm


def run_star_photometry(args):
    """
    Equivalent of the rsetstar script
    """
    nightshot = args.night + 'v' + args.shotid

    stars = get_shuffle_stars(args.shuffle_ifustars_dir, nightshot,
                              args.shuffle_mag_limit)

    curdir = os.path.abspath(os.path.curdir)

    for star in stars:

        stardir = curdir + '/%s_%d' % (nightshot, star.starid)
        os.mkdir(stardir)
        os.chdir(stardir)

        # Create the workdirectory for this star
        os.mkdir()

        # Extract data like the data in l1
        starobs, nshots = get_star_spectrum_data(star, args)

        # Call rspstar
        extract_star_spectrum(starobs, args)

        call_sumsplines(args.bin_dir, len(starobs))

        apply_factor_spline(nshots)

        call_fitonevp(args.bin_dir, args.extraction_wl,
                      nightshot+'_'+star.starid+'spec.res')

        # run_fit2d()


def prepare_model():
    """
2    Corresponds to rsp2 script
    """

    pass


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

    logging.info("Configuration {}.".format(args.config_source))

    vdrp_info = VdrpInfo.read(wdir)
    vdrp_info.night = args.night
    vdrp_info.shotid = args.shotid

    try:
        for task in tasks:
            os.chdir(wdir)
            run_star_photometry(args)
            # if task in ["cp_post_stamps", "all"]:
            #    # Copy over collapsed IFU cubes, aka IFU postage stamps.
            #    cp_post_stamps(wdir, args.reduction_dir, args.night,
            #                   args.shotid)

    finally:
        vdrp_info.save(wdir)
        logging.info("Done.")


if __name__ == "__main__":
    argv = None
    if argv is None:
        argv = sys.argv
    # Parse config file and command line paramters
    # command line parameters overwrite config file.
    args = parseArgs(argv)

    sys.exit(main(args))
