#!/usr/bin/env python
""" Photometry routine

Contains python translation of Karl Gebhardt

.. moduleauthor:: Jan Snigula <snigula@mpe.mpg.de>
"""

from __future__ import print_function
# import matplotlib

# from matplotlib import pyplot as plt

from argparse import RawDescriptionHelpFormatter as ap_RDHF
from argparse import ArgumentParser as AP

# import multiprocessing

import os
import shutil
import sys
import ConfigParser
import logging
import logging.config
import copy
from astropy.io import fits
# from astropy.io import ascii
import tempfile
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle

import vdrp.mplog as mplog
import vdrp.astrometry as astrom
import vdrp.programs as vp

from distutils import dir_util

import vdrp.utils as utils
from vdrp.mphelpers import MPPool, ThreadPool
from vdrp.vdrp_helpers import VdrpInfo, save_data, read_data, run_command
from vdrp.containers import DithAllFile


_baseDir = os.getcwd()

_logger = logging.getLogger()

# Parallelization code, we supply both a ThreadPool as well as a
# multiprocessing pool. Both start a given numer of threads/processes,
# that will work through the supplied tasks, till all are finished.
#
# The ThreadPool does not need to start subprocesses, but is limited by
# the Python Global Interpreter Lock (only one thread can access complex data
# types at one time). This can potentially slow things down.
#
# The MPPool needs to start up the processes, but this is only done once at
# the initializtion of the pool.
#
# The MPPool processes cannot start multiprocessing jobs themselves, so if
# you need nested parallelization, use the either ThreadPools for all, or
# Use one and the other.


class NoShotsException(Exception):
    pass


class Spectrum():
    """
    This class encapsulates the content of a tmp*.dat spectrum file

    Attributes
    ----------

    wl : float
        Wavelength
    cnts : float
        Counts of the spectrum
    flx : float
        Flux of the spectrum
    amp_norm : float
        Ampliflier normalization
    tp_norm : float
        Throughput normalization
    ftf_norm : float
        Fiber to fiber normalization
    err_cts : float

    err_cts_local : float

    err_max_flux : float

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
    """
    Class to store the information about one star from the shuffle output

    Attributes
    ----------

    starid : int
        ID for the star.
    shotid : int
        Shot number of the star observation
    shuffleid : int
        ID of the star in shuffle catalog
    ra : float
        Right ascension
    dec : float
        Declination
    catalog : str
        Catalog name used to find these.
    u : float
        U-Band magnitude from the shuffle catalog
    g : float
        G-Band magnitude from the shuffle catalog
    r : float
        R-Band magnitude from the shuffle catalog
    i : float
        I-Band magnitude from the shuffle catalog
    z : float
        Z-Band magnitude from the shuffle catalog
    """

    def __init__(self, starid='', shotid='', shuffleid=-1, ra=-1.0, dec=-1.0,
                 u=99., g=99., r=99., i=99., z=99., catalog='None'):
        self.starid = starid
        self.shotid = shotid
        self.shuffleid = shuffleid
        self.catalog = catalog
        self.ra = ra
        self.dec = dec
        self.mag_u = u
        self.mag_g = g
        self.mag_r = r
        self.mag_i = i
        self.mag_z = z


class StarObservation():
    """
    Data for one spectrum covering a star observation. This corresponds to the
    data stored in the l1 file with additions from other files

    Attributes
    ----------
    num : int
        Star number
    night : int
        Night of the observation
    shot : int
        Shot of the observation
    ra : float
        Right Ascension of the fiber center
    dec : float
        Declination of the fiber center
    x : float
        Offset of fiber relative to IFU center in x direction
    y : float
        Offset of fiber relative to IFU center in y direction
    full_fname : str
        Filename of the multi extension fits file.
    shotname : str
        NightvShot shot name
    expname : str
        Name of the exposure.
    dist : float
        Distance of the fiber from the star position
    offset_ra : float
        Offset in ra of the fiber from the star position
    offset_dec : float
        Offset in dec of the fiber from the star position
    fname : str
        Basename of the fits filenname
    ifuslot : str
        IFU slot ID
    avg : float
        Average of the spectrum
    avg_norm : float

    avg_error : float
        Error of the average of the spectrum
    structaz : float
        Azimuth of the telescope structure, read from the image header
    """
    def __init__(self, num=0., night=-1, shot=-1, ra=-1, dec=-1, x=-1, y=-1,
                 fname='', shotname='', expname='', offset_ra=-1,
                 offset_dec=1):

        self.num = 0
        self.night = -1.  # l1 - 10
        self.shot = -1.  # l1 - 11
        self.ra = -1.  # l1 - 1
        self.dec = -1.  # l1 - 2
        self.x = -1.  # l1 - 3
        self.y = -1.  # l1 - 4
        self.full_fname = ''  # l1 - 5
        self.shotname = ''  # l1 - 9
        self.expname = ''  # l1 - 6
        self.dist = -1.  # l1 - 7
        self.offset_ra = -1.
        self.offset_dec = -1.
        self.fname = ''
        self.ifuslot = ''

        # l1 - 8 is args.extraction_wl

        self.avg = 0.
        self.avg_norm = 0.
        self.avg_error = 0.

        self.structaz = -1.

    def set_fname(self, fname):
        """
        Split the full filename into the base name and the ifuslot
        """
        self.full_fname = fname
        self.fname, self.ifuslot = self.full_fname.split('.')[0].rsplit('_', 1)


def getDefaults():

    defaults = {}

    defaults["use_tmp"] = False
    defaults["remove_tmp"] = True

    defaults['photometry_logfile'] = 'photometry.log'

    defaults['shuffle_cores'] = 1

    defaults['starid'] = 1

    defaults['multi_shot'] = False
    # defaults['target_coords'] = False

    defaults['dithall_dir'] = '/work/00115/gebhardt/maverick/detect/'
    defaults["shuffle_mag_limit"] = 20.

    defaults["shuffle_ifustars_dir"] = \
        '/work/00115/gebhardt/maverick/sci/panacea/test/shifts/'
    defaults['multifits_dir'] = '/work/03946/hetdex/maverick/red1/reductions/'
    defaults['tp_dir'] = '/work/00115/gebhardt/maverick/detect/tp/'
    defaults['norm_dir'] = '/work/00115/gebhardt/maverick/getampnorm/all/'
    # defaults['bin_dir'] = '/home/00115/gebhardt/bin/'

    defaults['extraction_aperture'] = 1.6
    defaults['extraction_wl'] = 4505.
    defaults['extraction_wlrange'] = 1035.
    defaults['full_extraction_wl'] = 4500.
    defaults['full_extraction_wlrange'] = 1000.
    defaults['average_wl'] = 4500.
    defaults['average_wlrange'] = 10.
    defaults['radec_file'] = '/work/00115/gebhardt/maverick/getfib/radec.all'
    defaults['ifu_search_radius'] = 4.
    defaults['shot_search_radius'] = 600.

    # Shuffle parameters
    defaults["acam_magadd"] = 5.
    defaults["wfs1_magadd"] = 5.
    defaults["wfs2_magadd"] = 5.
    defaults["fplane_txt"] = "$config/fplane.txt"
    defaults["shuffle_cfg"] = "$config/shuffle.cfg"

    defaults['seeing'] = 1.5
    defaults['sdss_filter_file'] = \
        '/work/00115/gebhardt/maverick/detect/cal_script/sdssg.dat'

    defaults['sed_fit_dir'] = \
        '/work/00115/gebhardt/maverick/detect/sed/output/'
    defaults['sed_sigma_cut'] = 0.15
    defaults['sed_rms_cut'] = 0.01

    # Parameters for quick_fit
    defaults['quick_fit_ebv'] = 0.02
    defaults['quick_fit_plot'] = 0
    defaults['quick_fit_wave_init'] = 3540
    defaults['quick_fit_wave_final'] = 5540
    defaults['quick_fit_bin_size'] = 100

    defaults["task"] = "all"

    return defaults


def parseArgs(argv):
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
    args, remaining_argv = conf_parser.parse_known_args(argv)

    defaults = getDefaults()

    config_source = "Default"
    if args.conf_file:
        config_source = args.conf_file
        config = ConfigParser.SafeConfigParser()
        config.read([args.conf_file])
        defaults.update(dict(config.items("Photometry")))

        # bool_flags = ['use_tmp', 'remove_tmp', 'multi_shot', 'target_coords']
        bool_flags = ['use_tmp', 'remove_tmp', 'multi_shot']
        for bf in bool_flags:
            if config.has_option('Photometry', bf):
                defaults[bf] = config.getboolean('Photometry', bf)

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h

    # Inherit options from config_paarser
    parser = AP(parents=[conf_parser])

    parser.set_defaults(**defaults)
    parser.add_argument("--photometry_logfile", type=str,
                        help="Filename for log file.")

    parser.add_argument("--shuffle_cores", type=int,
                        help="Number of multiprocessing cores to use for"
                        "shuffle star extraction.")
    parser.add_argument("--starid", type=int,
                        help="Star ID to use, default is 1")
    parser.add_argument("--dithall_dir", type=str, help="Base directory "
                        "used to find the dithall.use files")
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
    # parser.add_argument("--bin_dir", type=str, help="Directory "
    #                     "with the fortran binary files.")

    parser.add_argument("--extraction_aperture", type=float, help="Aperture "
                        "radius in asec for the extraction")
    parser.add_argument("--extraction_wl", type=float, help="Central "
                        "wavelength for the extraction")
    parser.add_argument("--extraction_wlrange", type=float, help="Wavelength "
                        "range for the extraction")
    parser.add_argument("--full_extraction_wl", type=float, help="Central "
                        "wavelength for the full spectrum extraction")
    parser.add_argument("--full_extraction_wlrange", type=float,
                        help="Wavelength range for the full "
                        "spectrum extraction")
    parser.add_argument("--average_wl", type=float, help="Central "
                        "wavelength for the averaging")
    parser.add_argument("--average_wlrange", type=float, help="Wavelength "
                        "range for the averaging")
    parser.add_argument("--radec_file", type=str, help="Filename of file with "
                        "RA DEC PA positions for all shots")
    parser.add_argument("--ifu_search_radius", type=float, help="Radius for "
                        "search for fibers near a given star.")
    parser.add_argument("--shot_search_radius", type=float, help="Radius for "
                        "search for shots near a given star.")

    parser.add_argument("--seeing", type=float, help="Seeing in arcseconds"
                        " to assume for spectral extraction.")

    parser.add_argument("--target_ra", type=float, help="Target RA for multi"
                        " shot mode.")
    parser.add_argument("--target_dec", type=float, help="Target DEC for multi"
                        " shot mode.")

    parser.add_argument("--sdss_filter_file", type=str, help="Filter curve "
                        "for SDSS g-Band filter.")

    parser.add_argument("--sed_fit_dir", type=str, help="Directory with SED  "
                        "fit results.")
    parser.add_argument("--sed_sigma_cut", type=float, help="Sigma cut level"
                        " for combsed.")
    parser.add_argument("--sed_rms_cut", type=str, help="RMS cut level"
                        " for combsed.")

    # Parameters for quick_fit
    parser.add_argument("--quick_fit_ebv", type=float,
                        help="Extinction for star field")
    parser.add_argument("--quick_fit_plot", type=int,
                        help="Create SED fitting plots")
    parser.add_argument("--quick_fit_wave_init", type=float,
                        help="Initial wavelength for bin")
    parser.add_argument("--quick_fit_wave_final", type=float,
                        help="Final wavelength for bin")
    parser.add_argument("--quick_fit_bin_size", type=float,
                        help="Bin size for wavelength")

    parser.add_argument("-t", "--task", type=str, help="Task to execute.")

    # Shuffle parameters
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

    # Boolean paramters
    parser.add_argument("--use_tmp", action='store_true',
                        help="Use a temporary directory. Result files will"
                        " be copied to NIGHTvSHOT/res.")
    parser.add_argument("--multi_shot", action='store_true',
                        help="Run using all shots containing the star at the "
                        "given coordinates. Equivalent of rsp1 script")
    # parser.add_argument("--target_coords", action='store_true',
    #                     help="Run over all stars from shuffle for the given"
    #                     "night and shot, ignoring the ra and dec parameters")

    # positional arguments
    parser.add_argument('night', metavar='night', type=str,
                        help='Night of observation (e.g. 20180611).')
    parser.add_argument('shotid', metavar='shotid', type=str,
                        help='Shot ID (e.g. 017).')

    args = parser.parse_args(remaining_argv)

    args.config_source = config_source
    # should in principle be able to do this with accumulate???
    # args.use_tmp = args.use_tmp == "True"
    # args.remove_tmp = args.remove_tmp == "True"

    # NEW set the bin_dir to the vdrp supplied bin directory
    # args.bin_dir = utils.bindir()

    args.fplane_txt = utils.mangle_config_pathname(args.fplane_txt)
    args.shuffle_cfg = utils.mangle_config_pathname(args.shuffle_cfg)

    return args


def get_throughput_file(path, shotname):
    """
    Equivalent of rtp0 script.

    Checks if a night/shot specific throughput file exists.

    If true, return the filename, otherise the filename
    for an average throughput file.

    Parameters
    ----------
    path : str
        Path to the throughput files
    shotname : str
        Name of the shot
    """
    if os.path.exists(path + '/' + shotname + "sedtp_f.dat"):
        return path + '/' + shotname + "sedtp_f.dat"
    else:
        return path + '/' + "tpavg.dat"


def apply_factor_spline(factor, wdir):
    """
    Equivalent of the rawksp[12] scripts

    Apply the factor to the splines.out file. The factor is the number
    of individual shots the star was observed in.

    Parameters
    ----------
    factor : int
        The factor to apply.
    wdir : str
        Name of the work directory
    """
    wave, flx = np.loadtxt(os.path.join(wdir, 'splines.out'), unpack=True,
                           usecols=[0, 2])

    with open(os.path.join(wdir, 'fitghsp.in'), 'w') as f:
        for w, fl in zip(wave, flx):
            f.write('%f %f\n' % (w, fl*1.e17 / factor))


def get_star_spectrum_data(ra, dec, args, multi_shot=False, dithall=None):
    """
    This extracts the data about the different observations of the same star
    on different ifus.

    This is essentially the information stored in the l1 file.

    Parameters
    ----------
    ra : float
        Right Ascension of the star.
    dec : float
        Declination of the star.
    args : struct
        The arguments structure
    """

    if multi_shot:
        # First find matching shots
        _logger.info('Reading radec file %s' % args.radec_file)

        night, shot = np.loadtxt(args.radec_file, unpack=True, dtype='U50',
                                 usecols=[0, 1])
        ra_shot, dec_shot = np.loadtxt(args.radec_file, unpack=True,
                                       usecols=[2, 3])

        _logger.info('Searching for shots within %f arcseconds of %f %f'
                     % (args.shot_search_radius, ra, dec))
        # First find shots overlapping with the RA/DEC coordinates
        w_s = np.where(((np.sqrt((np.cos(dec/57.3)*(ra_shot-ra))**2
                                 + (dec_shot-dec)**2)*3600.)
                        < args.shot_search_radius))[0]

        if not len(np.where(w_s)[0]):
            raise NoShotsException('No shots found!')

        night = night[w_s]
        shot = shot[w_s]

    else:
        night = [args.night]
        shot = [args.shotid]

    night_shots = []
    starobs = []
    c = 0

    _logger.info('Found %d shots' % len(shot))

    for n, s in zip(night, shot):
        if multi_shot or dithall is None:
            dithall_file = args.dithall_dir+'/'+n+'v'+s+'/dithall.use'
            _logger.info('Reading dithall file %s' % dithall_file)
            try:
                dithall = DithAllFile(dithall_file)

            except Exception as e:
                _logger.warn('Failed to read %s' % dithall_file)
                _logger.exception(e)
                continue

        _logger.info('Filtering dithall file')
        filtered = dithall.where(((np.sqrt((np.cos(dec/57.3)
                                            * (dithall.ra-ra))**2
                                           + (dithall.dec-dec)**2) * 3600.)
                                  < args.ifu_search_radius))

        _logger.info('Found %d fibers' % len(filtered))

        for d in filtered:

            so = StarObservation()

            so.num = c+101
            so.night = n
            so.shot = s
            so.ra = d.ra
            so.dec = d.dec
            so.x = d.x
            so.y = d.y
            so.set_fname(d.filename)
            so.shotname = d.timestamp
            so.expname = d.expname

            so.dist = 3600.*np.sqrt((np.cos(dec/57.3)*(so.ra-ra))**2
                                    + (so.dec-dec)**2)

            # This is written to loffset
            so.offsets_ra = 3600.*(d.ra-ra)
            so.offsets_dec = 3600.*(d.dec-dec)

            # Make sure we actually have data for this shot
            fpath = '%s/%s/virus/virus%07d/%s/virus/%s' \
                % (args.multifits_dir, so.night, int(so.shot),
                   so.expname, so.fname) + '.fits'

            if not os.path.exists(fpath):
                _logger.warn('No fits data found for ifuslot %s in  %sv%s'
                             % (so.ifuslot, so.night, so.shot))
                continue

            starobs.append(so)
            night_shots.append('%s %s' % (n, s))

            c += 1

    return starobs, np.unique(night_shots)


def extract_star_spectrum(starobs, args, wl, wlr, wdir, prefix=''):
    """
    Equivalent of the rextsp[1] and parts of the rsp1b scripts.

    Extract stellar spectra, creating the tmp*.dat files. If prefix
    is set, it is prefixed to the tmp*dat file names.

    Parameters
    ----------
    starobs : list
        List with StarObservation objects.
    args : struct
        The arguments structure
    wdir : str
        Name of the work directory
    prefix : str (optional)
        Optional prefix for the output filenames.

    Returns
    -------
    list
        List of tmp*dat filenames created.
    """

    specfiles = []

    _logger.info('Extracting star spectrum')

    for s in starobs:
        fpath = '%s/%s/virus/virus%07d/%s/virus/%s' \
            % (args.multifits_dir, s.night, int(s.shot),
               s.expname, s.fname) + '.fits'
        vp.call_imextsp(fpath, s.ifuslot, wl, wlr,
                        get_throughput_file(args.tp_dir, s.night+'v'+s.shot),
                        args.norm_dir+'/'+s.fname+".norm",
                        prefix+'tmp%d.dat' % s.num, wdir)
        specfiles.append(prefix+'tmp%d.dat' % s.num)
    return specfiles


def get_shuffle_stars(nightshot, args, wdir):
    """
    Rerun shuffle and find the all stars for a given night / shot.

    Parameters
    ----------
    nightshot : str
        Night + shot name to work on.
    args : argparse.Namespace
        The script parameter namespace
    """

    astrom.get_ra_dec_orig(wdir, args.multifits_dir, args.night, args.shotid)
    track = astrom.get_track(wdir, args.multifits_dir, args.night, args.shotid)

    ra, dec, _ = \
        utils.read_radec(os.path.join(wdir, "radec.orig"))

    # Try to run shuffle using different catalogs, we need SDSS for SED
    # fitting, the others can be used for throughput calculation
    for cat in 'SDSS', 'GAIA', 'USNO':

        stars = []

        astrom.redo_shuffle(wdir, ra, dec, track,
                            args.acam_magadd, args.wfs1_magadd,
                            args.wfs2_magadd, args.shuffle_cfg,
                            args.fplane_txt, args.night, catalog=cat)

        c = 1
        try:
            indata_str = np.loadtxt(os.path.join(wdir, 'shout.ifustars'),
                                    dtype='U50', usecols=[0, 1])
            indata_flt = np.loadtxt(os.path.join(wdir, 'shout.ifustars'),
                                    dtype=float, usecols=[2, 3, 4, 5, 6, 7, 8])
            for ds, df in zip(indata_str, indata_flt):
                star = ShuffleStar(20000 + c, ds[0], ds[1], df[0], df[1],
                                   df[2], df[3], df[4], df[5], df[6], cat)
                if star.mag_g < args.shuffle_mag_limit:
                    stars.append(star)
                    c += 1

            if len(stars):
                return stars
            else:
                _logger.warn('No shuffle stars found using catalog %s' % cat)
        except OSError:
            _logger.warn('Failed to find shuffle stars for night %s, shot %s'
                         'using catalog %s' % (args.night, args.shotid, cat))

    _logger.error('No shuffle stars found at all!')


def average_spectrum(spec, wlmin, wlmax):
    """
    Corresponds to ravgsp0 script. Calculate the average of the
    spectrum in the range [wlmin, wlmax]

    Parameters
    ----------
    spec : Spectrum
        Spectrum class object
    wlmin : float
        Minimum wavelength of range to average.
    wlmax : float
        Maximum wavelength of range to average.

    Returns
    -------
    average, normaliztaion and uncertainty, equivalent to the spavg*.dat files.
    """

    wh = (spec.wl > wlmin) & (spec.wl < wlmax) & (spec.cnts != 0)

    # Calculate the mean of all values within wavelength range
    # where cnts are !=0

    if len(np.where(wh)[0]):
        avg = spec.cnts[wh].mean()
        norm = (spec.amp_norm[wh]*spec.tp_norm[wh]).mean()
        uncert = np.sqrt((spec.err_cts_local[wh]*spec.err_cts_local[wh]).sum()
                         / len(np.where(wh)[0]))
    else:
        avg = 0.
        norm = 0.
        uncert = 0.

    return avg, norm, uncert


def average_spectra(specfiles, starobs, wl, wlrange, wdir):
    """
    Average all observed spectra and fill in the corresponding entries in the
    StarObservation class.

    This corresponds to the ravgsp0 script

    Parameters
    ----------
    specfiles : list
        List of spectrum filenames.
    starobs : list
        List with StarObservation objects.
    wl : float
        Central wavelength for the averaging.
    wlrange : float
        Half width of the wavelength range for averaging.
    """

    wlmin = wl - wlrange
    wlmax = wl + wlrange

    with open(os.path.join(wdir, 'spavg.all'), 'w') as f:
        for spf, obs in zip(specfiles, starobs):
            sp = Spectrum()
            sp.read(os.path.join(wdir, spf))
            obs.avg, obs.avg_norm, obs.avg_error = \
                average_spectrum(sp, wlmin, wlmax)

            f.write('%f %.7f %.4f\n' % (obs.avg, obs.avg_norm, obs.avg_error))


def get_structaz(starobs, path):
    """
    Equivalent of the rgetadc script
    Read the STRUCTAZ parameter from the multi extension fits files and fill
    in the StarObservation entries.

    Parameters:
    -----------
    starobs : list
        List with StarObservation objects.
    path : string
        Path to the directory where the multi extension fits are stored.
    """
    missingobs = False
    az_vals = []
    m_obs = []

    for obs in starobs:
        fpath = '%s/%s/virus/virus%07d/%s/virus/%s' \
            % (path, obs.night, int(obs.shot),
               obs.expname, obs.fname) + '.fits'
        if not os.path.exists(fpath):
            missingobs = True
            m_obs.append(obs)
        else:
            with fits.open(fpath, 'readonly') as hdu:
                obs.structaz = hdu[0].header['STRUCTAZ']
                az_vals.append(obs.structaz)

    if missingobs and len(m_obs):  # Replace AZ values for missing fits images
        az_avg = np.average(az_vals)
        for obs in m_obs:
            obs.structaz = az_avg


def get_sedfits(starobs, args, wdir):
    """
    Run quick_fit to generate the SED fits, if available.

    If quick_fit cannot be imported, fall back to copying the files
    from sed_fit_dir

    Parameters:
    -----------
    starobss : list
        List with StarObservation objects.
    args : Namespace
        Namespace argument
    """

    try:
        import stellarSEDfits.quick_fit as qf
        from argparse import Namespace
        qf_args = Namespace()
        qf_args.filename = os.path.join(wdir, 'qf.ifus')
        qf_args.outfolder = wdir
        qf_args.ebv = args.quick_fit_ebv
        qf_args.make_plot = args.quick_fit_plot
        qf_args.wave_init = args.quick_fit_wave_init
        qf_args.wave_final = args.quick_fit_wave_final
        qf_args.bin_size = args.quick_fit_bin_size

        have_stars = False

        with open(os.path.join(wdir, 'qf.ifus'), 'w') as f:
            for s in starobs:
                if s.catalog != 'SDSS':
                    _logger.info('Skipping %s star %d, currently only SDSS'
                                 ' stars support SED fitting.'
                                 % (s.catalog, s.starid))
                else:
                    f.write('%s %s %f %f %f %f %f %f %f\n'
                            % (s.shotid, s.shuffleid, s.ra, s.dec, s.mag_u,
                               s.mag_g, s.mag_r, s.mag_i, s.mag_z))
                    have_stars = True

        if have_stars:
            qf.main(qf_args)
            for s in starobs:
                fitsedname = os.path.join(wdir, '%s_%s.txt'
                                          % (s.shotid, s.shuffleid))
                sedname = os.path.join(wdir, 'sp%d_fitsed.dat' % s.starid)
                if not os.path.exists(fitsedname):
                    _logger.warn('No sed fit found for star %d' % s.starid)
                    continue
                shutil.move(fitsedname, sedname)

    except ImportError:
        _logger.warn('Failed to import quick_fit, falling back to '
                     'pre-existing SED fits')
        for s in starobs:
            fitsedname = '%s_%s.txt' % (s.shotid, s.shuffleid)
            sedname = os.path.join('sp%d_fitsed.dat' % s.starid)
            if not os.path.exists(os.path.join(args.sed_fit_dir, fitsedname)):
                _logger.warn('No sed fit found for star %d' % s.starid)
                continue
            shutil.copy2(os.path.join(args.sed_fit_dir, fitsedname), sedname)


def run_fit2d(ra, dec, starobs, seeing, outname, wdir):
    """
    Prepare input files for running fit2d, and run it.

    Parameters
    ----------
    ra : float
        Right Ascension of the star.
    dec : float
        Declination of the star.
    starobs : list
        List with StarObservation objects.
    seeing : float
        Assumed seeing for the observation.
    outname : str
        Output filename.

    """
    with open(os.path.join(wdir, 'in'), 'w') as f:
        for obs in starobs:
            f.write('%f %f %f %f %s %s %s %s %f %f\n'
                    % (obs.ra, obs.dec, obs.avg, obs.avg_norm, obs.shotname,
                       obs.night, obs.shot, obs.expname, obs.structaz,
                       obs.avg_error))
    if not os.path.exists(os.path.join(wdir, 'fwhm.use')):
        _logger.warn('No fwhm from getnormexp found, using default')
        with open(os.path.join(wdir, 'fwhm.use'), 'w') as f:
            f.write('%f\n' % seeing)

    vp.call_fit2d(ra, dec, outname, wdir)


def run_sumlineserr(specfiles, wdir):
    """
    Prepare input and run sumlineserr. It sums a set of spectra, and then bins
    to 100AA bins. Used for SED fitting.

    Parameters
    ----------
    specfiles : list
        List of spectrum filenames.

    """

    indata = np.loadtxt(os.path.join(wdir, 'out2d'), dtype='U50', ndmin=2,
                        usecols=[8, 9, 10, 11, 12, 13, 14])

    with open(os.path.join(wdir, 'list2'), 'w') as f:
        for spf, d in zip(specfiles, indata):
            f.write('%s %s %s %s %s %s %s %s\n' %
                    (spf, d[0], d[1], d[2], d[3], d[4], d[5], d[6]))

    run_command(vp._vdrp_bindir + '/sumlineserr', wdir=wdir)


def run_fitem(wl, outname, wdir):
    """
    Prepare input file for fitem, and run it.

    Parameters
    ----------
    wl : float
        Wavelength
    outname : str
        Base output filename.

    Output
    ------
    outname+'spece.dat' :
        Saved input file.
    outname+'_2dn.ps' :
        Control plot
    outname+'_2d.res' :
        Parameters of the line fit
    """

    indata = np.loadtxt(os.path.join(wdir, 'splines.out'), dtype='U50',
                        usecols=[0, 1, 2, 3, 4])

    with open(os.path.join(wdir, 'fitghsp.in'), 'w') as f:
        for d in indata:
            f.write('%s %s %s %s %s\n' %
                    (d[0], d[2], d[4], d[1], d[3]))

    vp.call_fitem(wl, wdir)

    shutil.move(os.path.join(wdir, 'fitghsp.in'), outname+'spece.dat')
    shutil.move(os.path.join(wdir, 'pgplot.ps'), outname+'_2dn.ps')
    shutil.move(os.path.join(wdir, 'lines.out'), outname+'_2d.res')


def run_getsdss(filename, sdss_file, wdir):
    """
    Run getsdss on filename. Equivalent to rsdss file.

    Parameters
    ----------
    filename : str
        Filename with spectral data
    sdss_file : str
        Full path and filename to the sdss g-band filter curve.

    Returns
    -------
    The flux in the g-Band.

    """
    shutil.copy2(os.path.join(wdir, sdss_file),
                 os.path.join(wdir, 'sdssg.dat'))
    shutil.copy2(os.path.join(wdir, filename), os.path.join(wdir, 's1'))

    run_command(vp._vdrp_bindir + '/getsdssg', wdir=wdir)

    return float(np.loadtxt(os.path.join(wdir, 'out')))


def run_biwt(data, outfile, wdir):
    """
    Calculate biweight of the supplied data.

    Parameters
    ----------
    data : list
        List of the data to be run through biwt.

    Returns
    -------
    n, biwt, error
    """
    with open(os.path.join(wdir, 'tp.dat'), 'w') as f:
        for d in data:
            f.write('%f\n' % d)

    run_command(vp._vdrp_bindir + '/biwt', 'tp.dat\n1\n', wdir=wdir)

    os.remove(os.path.join(wdir, 'tp.dat'))

    shutil.move(os.path.join(wdir, 'biwt.out'), os.path.join(wdir, outfile))


def run_combsed(sedlist, sigmacut, rmscut, outfile, wdir, plotfile=None):
    """


    Parameters
    ----------
    sedlist : list
        List of filenames of SED fits
    sigmacut : float
        Cut value for sigma
    rmscut : float
        Cut value for rms
    outfile : str
        Output filename
    plotfile : str (optional)
        Optional plot output filename

    Returns
    -------
    n, biwt, error
    """
    with open(os.path.join(wdir, 'list'), 'w') as f:
        for l in sedlist:
            f.write('%s\n' % l)

    input = '{:f} {:f}\n'
    run_command(vp._vdrp_bindir + '/combsed',
                input.format(sigmacut, rmscut), wdir=wdir)

    shutil.move(os.path.join(wdir, 'comb.out'), os.path.join(wdir, outfile))

    if plotfile is not None:

        fdata = np.loadtxt(os.path.join(wdir, 'out'), dtype=float,
                           usecols=[1, 2, 4, 5])
        idata = np.loadtxt(os.path.join(wdir, 'out'), dtype=int,
                           usecols=[0, 3])

        f_in = os.path.join(wdir, 'in')
        f_in2 = os.path.join(wdir, 'in2')
        with open(f_in, 'w') as f, open(f_in2, 'w') as f2:
            for di, df, sf in zip(idata, fdata, sedlist):
                if di[1] == 0:
                    f.write('%s %d %f %f %d %f %f\n'
                            % (sf, di[0], df[0], df[1],
                               di[1], df[2], df[3]))
                    f2.write('%s %d %f %f %d %f %f\n'
                             % (sf, di[0], df[0], df[1],
                                di[1], df[2], df[3]))
            f2.write('%s\n' % outfile)

        run_command(vp._vdrp_bindir + '/plotseda', '/vcps\n', wdir=wdir)

        shutil.move(os.path.join(wdir, 'pgplot.ps'),
                    os.path.join(wdir, plotfile))


def copy_stardata(starname, starid, wdir):
    """
    Copies the result files from workdir results_dir as done by rspstar.

    Parameters
    ----------
    starname : str
        Star name to copy over.
    starid : int
        Star ID to use for the final filename.
    results_dir : str
        Final directory for results.

    """

    shutil.copy2(os.path.join(wdir, starname+'specf.dat'),
                 os.path.join(wdir, 'sp%d_2.dat' % starid))
    shutil.copy2(os.path.join(wdir, 'sumspec.out'),
                 os.path.join(wdir, 'sp%d_100.dat' % starid))


def run_shuffle_photometry(args, wdir):
    """
    Equivalent of the rsetstar script. Find all shuffle stars observed
    for the night / shot given on the command line, and the loop over all
    stars ra / dec.

    Parameters
    ----------
    args : struct
        The arguments structure

    """
    nightshot = args.night + 'v' + args.shotid

    stars = get_shuffle_stars(nightshot, args, wdir)

    # Parallelize the star extraction. Create a MPPool with
    # shuffle_cores processes

    pool = MPPool(args.jobnum, args.shuffle_cores)

    for star in stars:

        # Add all the tasks, they will start right away.
        pool.add_task(run_star_photometry, nightshot, star.ra, star.dec,
                      star.starid, copy.copy(args))

    # Now wait for all tasks to finish.
    pool.wait_completion()

    _logger.info('Saving star data for %s' % nightshot)

    save_data(stars, os.path.join(args.results_dir, '%s.shstars' % nightshot))


def run_star_photometry(nightshot, ra, dec, starid, args):
    """
    Equivalent of the rsp1a2b script.

    Run the stellar extraction code for a given ra / dec position.

    Parameters
    ----------
    ra : float
        Right Ascension of the star.
    dec : float
        Declination of the star.
    starid : int
        ID to give to the star / position
    args : struct
        The arguments structure

    """
    try:
        _logger.info('Starting star extraction')
        nightshot = args.night + 'v' + args.shotid

        starname = '%s_%d' % (nightshot, starid)

        _logger.info('Extracting star %s' % starname)

        # Create the workdirectory for this star
        # curdir = os.path.abspath(os.path.curdir)
        curdir = args.wdir
        stardir = os.path.join(curdir, starname)
        if not os.path.exists(stardir):
            os.mkdir(stardir)

        # Extract data like the data in l1
        starobs, nshots = get_star_spectrum_data(ra, dec, args,
                                                 args.multi_shot)

        if not len(starobs):
            _logger.warn('No shots found, skipping!')
            return

        # Call rspstar
        # Get fwhm and relative normalizations
        vp.call_getnormexp(nightshot, stardir)

        specfiles = extract_star_spectrum(starobs, args,
                                          args.extraction_wl,
                                          args.extraction_wlrange,
                                          stardir)

        vp.call_sumsplines(len(starobs), stardir)

        apply_factor_spline(len(nshots), stardir)

        vp.call_fitonevp(args.extraction_wl, nightshot+'_'+str(starid),
                         stardir)

        average_spectra(specfiles, starobs, args.average_wl,
                        args.average_wlrange, stardir)

        get_structaz(starobs, args.multifits_dir)

        run_fit2d(ra, dec, starobs, args.seeing, starname + '.ps', stardir)

        # Save the out2 file created by fit2d
        shutil.copy2(os.path.join(stardir, 'out2'),
                     os.path.join(stardir, 'sp%d_out2.dat') % starid)

        vp.call_mkimage(ra, dec, starobs, stardir)

        run_sumlineserr(specfiles, stardir)

        run_fitem(args.extraction_wl, starname, stardir)

        # Extract full spectrum

        fspecfiles = extract_star_spectrum(starobs, args,
                                           args.full_extraction_wl,
                                           args.full_extraction_wlrange,
                                           stardir, prefix='f')

        run_sumlineserr(fspecfiles, stardir)

        indata = np.loadtxt(os.path.join(stardir, 'splines.out'), dtype='U50',
                            usecols=[0, 1, 2, 3, 4])

        with open(os.path.join(stardir, starname + 'specf.dat'), 'w') as f:
            for d in indata:
                f.write('%s %s %s %s %s\n' % (d[0], d[2], d[4], d[1], d[3]))

        vp.call_sumspec(starname, stardir)

        mind = args.shot_search_radius
        for o in starobs:
            if o.dist < mind:
                mind = o.dist

        _logger.info('Closest fiber is %.5f arcseconds away' % mind)

        copy_stardata(starname, starid, stardir)

        _logger.info('Saving star data for %d' % starid)
        save_data(starobs, os.path.join(stardir, 'sp%d.obsdata' % starid))

        # Finally save the results to the results_dir

        _logger.info('Saving data for %s' % starname)
        shutil.copy2(os.path.join(stardir, starname+'.ps'), args.results_dir)
        shutil.copy2(os.path.join(stardir, starname+'_2d.res'),
                     args.results_dir)
        shutil.copy2(os.path.join(stardir, starname+'_2dn.ps'),
                     args.results_dir)
        shutil.copy2(os.path.join(stardir, starname+'spec.dat'),
                     args.results_dir)
        shutil.copy2(os.path.join(stardir, starname+'spec.res'),
                     args.results_dir)
        shutil.copy2(os.path.join(stardir, starname+'spece.dat'),
                     args.results_dir)
        shutil.copy2(os.path.join(stardir, starname+'specf.dat'),
                     args.results_dir)
        shutil.copy2(os.path.join(stardir, starname+'tot.ps'),
                     args.results_dir)
        shutil.copy2(os.path.join(stardir, 'sp%d_2.dat') % starid,
                     args.results_dir)
        shutil.copy2(os.path.join(stardir, 'sp%d_100.dat') % starid,
                     args.results_dir)
        shutil.copy2(os.path.join(stardir, 'sp%d.obsdata') % starid,
                     args.results_dir)
        shutil.copy2(os.path.join(stardir, 'sp%d_out2.dat') % starid,
                     args.results_dir)

        _logger.info('Finished star extraction for %s' % starname)
    except Exception as e:
        _logger.exception(e)


def get_g_band_throughput(args):
    '''
    Measure the throughput in the SDSS g-Band
    Equivalent of the rgettp0 script

    Parameters
    ----------
    args : struct
        The arguments structure

    '''

    nightshot = args.night + 'v' + args.shotid

    wdir = args.wdir

    _logger.info('Reading %s.shstars in %s'
                 % (nightshot, wdir))

    stars = read_data(os.path.join(wdir, '%s.shstars') % nightshot)

    flxdata = []

    for s in stars:
        if not os.path.exists(os.path.join(wdir, 'sp%d.obsdata' % s.starid)):
            _logger.warn('No spectral data for %d found!' % s.starid)
            continue
        starobs = read_data(os.path.join(wdir, 'sp%d.obsdata' % s.starid))
        g_flx = run_getsdss('sp%d_100.dat' % s.starid, args.sdss_filter_file,
                            wdir)
        sdss_flx = 5.048e-9*(10**(-0.4*s.mag_g))/(6.626e-27) / \
            (3.e18/4680.)*360.*5.e5*100

        dflx = 0.
        if sdss_flx > 0.:
            dflx = g_flx / sdss_flx

        if len(starobs) > 15 and dflx > 0.02 and dflx < 0.21:
            flxdata.append(dflx)

    run_biwt(flxdata, 'tp.biwt', wdir)


def mk_sed_throughput_curve(args):
    '''
    Equivalent of the rgett0b script.

    Parameters
    ----------
    args : struct
        The arguments structure

    '''

    nightshot = args.night + 'v' + args.shotid

    wdir = args.wdir

    _logger.info('Reading %s.shstars in %s'
                 % (nightshot, wdir))

    stars = read_data(os.path.join(wdir, '%s.shstars' % nightshot))

    sedlist = []

    get_sedfits(stars, args, wdir)

    for s in stars:
        if not os.path.exists(os.path.join(wdir, 'sp%s_100.dat' % s.starid)):
            _logger.info('No star data found for sp%s_100.dat' % s.starid)
            continue
        sedname = os.path.join(wdir, 'sp%d_fitsed.dat' % s.starid)

        if not os.path.exists(sedname):
            _logger.warn('No sed fit found for star %d' % s.starid)
            continue

        seddata = np.loadtxt(sedname, ndmin=1).transpose()
        # stardata = np.loadtxt('sp%s_100.dat' % s.starid).transpose()

        sedcgs = seddata[1][1:]/6.626e-27/(3.e18/seddata[0][1:])*360.*5.e5*100.

        np.savetxt(os.path.join(wdir, 'sp%dsed.dat' % s.starid),
                   zip(seddata[0][1:], seddata[1][1:]/sedcgs))

        sedlist.append('sp%dsed.dat' % s.starid)

    if not len(sedlist):
        _logger.warn('No SED fits found, skipping SED throughput curve '
                     'generation')
        return

    run_combsed(sedlist, args.sed_sigma_cut, args.sed_rms_cut,
                '%ssedtp.dat' % nightshot, wdir, '%ssedtpa.ps' % nightshot)

    data = np.loadtxt(os.path.join(wdir, '%ssedtp.dat' % nightshot))

    with open(os.path.join(wdir, '%sfl.dat' % nightshot), 'w') as f:
        for d in data:
            f.write('%f %f\n' % (d[0], 6.626e-27*(3.e18/d[0])/360.
                                 / 5.e5/d[1]*250.))

    with open(os.path.join(wdir, 'offsets.dat'), 'w') as off:
        for star in stars:
            use_star = False
            if not os.path.exists(os.path.join(wdir, 'sp%s_100.dat'
                                               % star.starid)):
                _logger.debug('sp%s_100.dat not found!' % star.starid)
                continue
            with open(os.path.join(wdir, 'sp%s_100.dat'
                                   % star.starid), 'r') as f:
                for l in f.readlines():
                    w, v = l.split()
                    if w.strip().startswith('4540') and float(v) > 10000.:
                        use_star = True
                        break
            if use_star:
                with open(os.path.join(wdir, 'sp%d_out2.dat'
                                       % star.starid), 'r') as f:
                    line = f.readline()
                    vals = line.split()
                    if float(vals[3]) > -0.5 and float(vals[3]) < 0.5 and \
                       float(vals[4]) > -0.5 and float(vals[4]) < 0.5:
                        off.write('%d %s' % (star.starid, line))


vdrp_info = None


def main(jobnum, args):
    """
    Main function.
    """
    global vdrp_info

    # Create results directory for given night and shot
    cwd = _baseDir
    results_dir = os.path.join(cwd, args.night + 'v' + args.shotid,  'res')
    utils.createDir(results_dir)
    args.results_dir = results_dir

    # save arguments for the execution
    with open(os.path.join(results_dir, 'args.pickle'), 'wb') as f:
        pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)

    tasks = args.task.split(",")
    if args.use_tmp and tasks != ['all'] and tasks != ['extract_coord']:
        _logger.error("Step-by-step execution not possible when running "
                      "in a tmp directory.")
        _logger.error("   Please either call without -t or set "
                      "use_tmp to False.")
        sys.exit(1)

    _logger.info("Executing tasks : {}".format(tasks))

    # default is to work in results_dir
    wdir = results_dir
    if args.use_tmp:
        # Create a temporary directory
        tmp_dir = tempfile.mkdtemp()
        _logger.info("Tempdir is {}".format(tmp_dir))
        _logger.info("Copying over prior data (if any)...")
        dir_util.copy_tree(results_dir, tmp_dir)
        # set working directory to tmp_dir
        wdir = tmp_dir

    _logger.info("Configuration {}.".format(args.config_source))

    vdrp_info = VdrpInfo.read(wdir)
    vdrp_info.night = args.night
    vdrp_info.shotid = args.shotid

    args.curdir = os.path.abspath(os.path.curdir)
    args.wdir = wdir
    args.jobnum = jobnum

    try:
        for task in tasks:
            # os.chdir(wdir)

            if task in ['extract_coord']:
                # Equivalent of rsp1
                _logger.info('Running on a single RA/DEC position')
                if args.target_ra is None or args.target_dec is None:
                    raise Exception('To run on a specific position, please '
                                    'specify target_ra and target_dec of the'
                                    ' position')
                run_star_photometry(args.target_ra, args.target_dec,
                                    args.starid, args)

            if task in ['extract_stars', 'all']:
                # os.chdir(wdir)
                # Equivalent of rsetstar
                _logger.info('Extracting all shuffle stars')
                run_shuffle_photometry(args, wdir)
                _logger.info('Finished star extraction')
            if task in ['get_g_band_throughput', 'all']:
                # os.chdir(wdir)
                _logger.info('Getting g-band photometry')
                get_g_band_throughput(args)

            if task in ['mk_sed_throughput_curve', 'all']:
                # os.chdir(wdir)
                _logger.info('Creating SED throughput curve')
                mk_sed_throughput_curve(args)

            if task in ['fit_throughput_curve', 'all']:
                pass
    except Exception as e:
        _logger.exception(e)

    finally:
        # os.chdir(args.curdir)
        vdrp_info.save(wdir)
        _logger.info("Done.")


def run():
    argv = None
    if argv is None:
        argv = sys.argv

    # Here we create another external argument parser, this checks if we
    # are supposed to run in multi-threaded mode.

    # First check if we should loop over an input file
    parser = AP(description='Test', formatter_class=ap_RDHF, add_help=False)
    # parser.add_argument('args', nargs=ap_remainder)
    parser.add_argument('-M', '--multi', help='Input filename to loop over, '
                        'append a range in the format [min:max] to select a '
                        'subsection of the lines')
    parser.add_argument('--mcores', type=int, default=1,
                        help='Number of paralles process to execute.')
    parser.add_argument('-l', '--logfile', type=str, default='vdrp.log',
                        help='Logfile to write to.')

    args, remaining_argv = parser.parse_known_args()

    # Setup the logging
    fmt = '%(asctime)s %(levelname)-8s %(threadName)12s %(funcName)15s(): ' \
        '%(message)s'
    formatter = logging.Formatter(fmt, datefmt='%m-%d %H:%M:%S')
    _logger.setLevel(logging.DEBUG)

    cHndlr = logging.StreamHandler()
    cHndlr.setLevel(logging.DEBUG)
    cHndlr.setFormatter(formatter)

    _logger.addHandler(cHndlr)

    fHndlr = logging.FileHandler(args.logfile, mode='w')
    fHndlr.setLevel(logging.DEBUG)
    fHndlr.setFormatter(formatter)

    _logger.addHandler(fHndlr)

    # Wrap the log handlers with the MPHandler, this is essential for the use
    # of multiprocessing, otherwise, tasks will hang.
    mplog.install_mp_handler(_logger)

    # We found a -M flag with a command file, now loop over it, we parse
    # the command line parameters for each call, and intialize the args
    # namespace for this call.
    if args.multi:
        mfile = args.multi.split('[')[0]

        if not os.path.isfile(mfile):
            raise Exception('%s is not a file?' % mfile)

        try:  # Try to read the file
            with open(mfile) as f:
                cmdlines = f.readlines()
        except Exception as e:
            print(e)
            raise Exception('Failed to read input file %s!' % args.multi)

        # Parse the line numbers to evaluate, if any given.
        if args.multi.find('[') != -1:
            try:
                minl, maxl = args.multi.split('[')[1].split(']')[0].split(':')
            except ValueError:
                raise Exception('Failed to parse line range, should be of '
                                'form [min:max]!')

            cmdlines = cmdlines[int(minl):int(maxl)]

        # Create the ThreadPool.
        pool = ThreadPool(args.mcores)
        c = 1

        # For each command line add an entry to the ThreadPool.
        for l in cmdlines:
            largs = copy.copy(remaining_argv)
            largs += l.split()

            main_args = parseArgs(largs)

            pool.add_task(main, c, copy.copy(main_args))

        # Wait for all tasks to complete
        pool.wait_completion()

        sys.exit(0)
    else:
        # Parse config file and command line paramters
        # command line parameters overwrite config file.

        # The first positional argument wasn't an input list,
        # so process normally
        args = parseArgs(remaining_argv)

        sys.exit(main(1, args))


if __name__ == "__main__":
    run()
