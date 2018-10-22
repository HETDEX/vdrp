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

import time
import multiprocessing
import threading
import Queue

import os
import shutil
import sys
import ConfigParser
import logging
import logging.config
import copy
import multiprocessing.pool
import subprocess
from astropy.io import fits
# from astropy.io import ascii
import tempfile
import numpy as np
from collections import OrderedDict
try:
    import cPickle as pickle
except ImportError:
    import pickle

import mplog

from distutils import dir_util

import utils

_masterLock = threading.RLock()
_baseDir = os.getcwd()

_logger = logging.getLogger()


class ThreadWorker(threading.Thread):
    """Thread executing tasks from a given tasks queue"""
    def __init__(self, name, tasks):
        threading.Thread.__init__(self)
        self.name = name
        self.tasks = tasks
        self.daemon = True
        self.start()

    def run(self):
        threading.current_thread().name = self.name
        while True:
            try:
                func, args, kargs = self.tasks.get(True, 2.0)
                try:
                    func(*args, **kargs)
                except Exception as e:
                    print(e)
                finally:
                    self.tasks.task_done()
            except Queue.Empty:
                print('%s %s queue is empty, shutting down!'
                      % (time.strftime('%H:%M:%S'), self.name))
                return
            except Exception as e:
                print(e)


class MPWorker(multiprocessing.Process):
    """Thread executing tasks from a given tasks queue"""
    def __init__(self, name, tasks):
        multiprocessing.Process.__init__(self)
        self.name = name
        self.tasks = tasks
        self.start()

    def run(self):
        while True:
            try:
                func, args, kargs = self.tasks.get(True, 2.0)
                try:
                    func(*args, **kargs)
                except Exception as e:
                    print(e)
                finally:
                    self.tasks.task_done()
            except Queue.Empty:
                print('%s %s queue is empty, shutting down!'
                      % (time.strftime('%H:%M:%S'), self.name))
                return
            except Exception as e:
                print(e)


class ThreadPool:
    """Pool of threads consuming tasks from a queue"""
    def __init__(self, num_threads):
        self.tasks = Queue.Queue(num_threads)
        for i in range(num_threads):
            ThreadWorker('ThreadWorker%d' % i, self.tasks)

    def add_task(self, func, *args, **kargs):
        """Add a task to the queue"""
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        """Wait for completion of all the tasks in the queue"""
        self.tasks.join()


class MPPool:
    """Pool of threads consuming tasks from a queue"""
    def __init__(self, jobnum, num_proc):
        self.tasks = multiprocessing.JoinableQueue(num_proc)
        for i in range(num_proc):
            MPWorker('MPWorker%d_%d' % (jobnum, i), self.tasks)

    def add_task(self, func, *args, **kargs):
        """Add a task to the queue"""
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        """Wait for completion of all the tasks in the queue"""
        self.tasks.join()


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

    def __init__(self, starid=-1, shotid=-1, shuffleid=-1, ra=-1.0, dec=-1.0,
                 u=99., g=99., r=99., i=99., z=99.):
        self.starid = starid
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

    defaults = {}

    defaults["use_tmp"] = False
    defaults["remove_tmp"] = True

    defaults['photometry_logfile'] = 'photometry.log'

    defaults['starid'] = 1

    defaults['multi_shot'] = False
    defaults['target_coords'] = False

    defaults['dithall_dir'] = '/work/00115/gebhardt/maverick/detect/'
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
    defaults['average_wlrange'] = 100.
    defaults['radec_file'] = '/work/00115/gebhardt/maverick/getfib/radec.all'
    defaults['ifu_search_radius'] = 4.
    defaults['shot_search_radius'] = 600.

    defaults['seeing'] = 1.5
    defaults['sdss_filter_file'] = \
        '/work/00115/gebhardt/maverick/detect/cal_script/sdssg.dat'

    defaults["task"] = "all"

    config_source = "Default"
    if args.conf_file:
        config_source = args.conf_file
        config = ConfigParser.SafeConfigParser()
        config.read([args.conf_file])
        defaults.update(dict(config.items("Photometry")))

        bool_flags = ['use_tmp', 'remove_tmp', 'multi_shot', 'target_coords']
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
    parser.add_argument("--bin_dir", type=str, help="Directory "
                        "with the fortran binary files.")

    parser.add_argument("--extraction_aperture", type=float, help="Aperture "
                        "radius in asec for the extraction")
    parser.add_argument("--extraction_wl", type=float, help="Central "
                        "wavelength for the extraction")
    parser.add_argument("--extraction_wlrange", type=float, help="Wavelength "
                        "range for the extraction")
    parser.add_argument("--average_wlrange", type=float, help="Wavelength "
                        "range for the averaging")
    parser.add_argument("--radec_file", type=str, help="Filename of file with "
                        "RA DEC PA positions for all shots")
    parser.add_argument("--ifu_search_radius", type=float, help="Radius for "
                        "search for fibers near a given star.")
    parser.add_argument("--shot_search_radius", type=float, help="Radius for "
                        "search for shots near a given star.")

    parser.add_argument("--seeing", type=float, help="Seeing in arcseconds"
                        "to assume for spectral extraction.")

    parser.add_argument("--target_ra", type=float, help="Target RA for multi"
                        " shot mode.")
    parser.add_argument("--target_dec", type=float, help="Target DEC for multi"
                        " shot mode.")

    parser.add_argument("--sdss_filter_file", type=str, help="Filter cureve "
                        "for SDSS g-Band filter.")

    parser.add_argument("-t", "--task", type=str, help="Task to execute.")

    # Boolean paramters
    parser.add_argument("--use_tmp", action='store_true',
                        help="Run using all shots containing the star at the "
                        "given coordinates. Equivalent of rsp1 script")
    parser.add_argument("--multi_shot", action='store_true',
                        help="Run using all shots containing the star at the "
                        "given coordinates. Equivalent of rsp1 script")
    parser.add_argument("--target_coords", action='store_true',
                        help="Run over all stars from shuffle for the given"
                        "night and shot, ignoring the ra and dec parameters")

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

    return args


def save_data(d, filename):
    # save data for later tasks
    with open(filename, 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)


def read_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def run_command(cmd, input=None):
    """
    Run and fortran command sending the optional input string on stdin.

    Parameters
    ----------
    cmd : str
        The command to be run, must be full path to executable
    input : str, optional
        Input to be sent to the command through stdin.
    """
    _logger.info('Running %s' % cmd)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    so, _ = proc.communicate(input=input)
    for l in so.split("\n"):
        _logger.info(l)
    proc.wait()


def call_imextsp(bindir, filename, ifuslot, wl, wlw, tpavg, norm, outfile):
    """
    Equivalent of the rextsp script,
    a wrapper around the imextsp fortran routine.

    Extracts the spectrum from the multi fits files and writes the tmp*dat.
    This also calculates the appropriate photon errors, using counting and
    sky residual errors. This applies the throughput and fiber to fiber.

    Parameters
    ----------
    bindir : str
        The path to the imextsp binary
    filename : str
        The filename to process
    ifuslot : str
        The ifuslot name
    wl : float
        The central extraction wavelength
    wlw : float
        The width of the extraction window around wl
    tpavg : float
        Throughput average for the spectrum
    norm : float
        Fiber to fiber normaliztion for the spectrum
    outfile : str
        Name of the output filename
    """
    input = '"{filename:s}"\n{ifuslot} {wl} {wlw}\n"{tpavg}"\n"{norm}"\n'

    try:
        os.remove('out.sp')
    except OSError:
        pass

    try:
        os.remove(outfile)
    except OSError:
        pass

    s = input.format(filename=filename, ifuslot=ifuslot, wl=wl, wlw=wlw,
                     tpavg=tpavg, norm=norm)

    run_command(bindir + '/imextsp', s)

    shutil.move('out.sp', outfile)


def call_sumsplines(bindir, nspec):
    """
    Call sumsplines, calculate a straight sum of the spectra in a list,
    including errors. Expects the spectra to be called tmp101 to
    tmp100+nspec.

    Creates a file called splines.out

    Parameters
    ----------
    bindir : str
        The path to the sumsplines binary
    nspec : int
        Number of spectra to read.
    """
    with open('list', 'w') as f:
        for i in range(0, nspec):
            f.write('tmp{c}.dat\n'.format(c=i+101))

    run_command(bindir + '/sumsplines')


def call_fitonevp(bindir, wave, outname):
    """
    Call fitonevp

    Requires fitghsp.in created by apply_factor_spline

    Parameters
    ----------
    bindir : str
        The path to the sumsplines binary
    wave : float
        Wavelength
    outname : str
        Output filename
    """
    input = '0 0\n{wave:f}\n/vcps\n'

    run_command(bindir + 'fitonevp', input.format(wave=wave))

    shutil.move('pgplot.ps', outname+'tot.ps')
    shutil.move('out', outname+'spec.dat')
    shutil.move('lines.out', outname+'spec.res')

    splinedata = np.loadtxt('splines.out')

    with open(outname+'spece.dat', 'w') as f:
        for d in splinedata:
            f.write('%.4f\t%.7f\t%.8e\t%.7f\t%.8e\n'
                    % (d[0], d[1], d[3], d[2]*1e17, d[4]*1e17))


def call_fit2d(bindir, ra, dec, outname):
    """
    Call fit2d. Calculate the 2D spatial fit based on fwhm, fiber locations,
    and ADC. This convolves the PSF over each fiber, for a given input
    position. It fits the ampltiude, minimizing to a chi^2.

    Requires input files generated by run_fit2d

    Parameters
    ----------
    bindir : str
        The path to the fit2d binary
    ra : float
        Right Ascension of the star.
    dec : float
        Declination of the star.
    outname : str
        Output filename.
    """
    input = '{ra:f} {dec:f}\n/vcps\n'

    run_command(bindir + '/fit2d', input.format(ra=ra, dec=dec))

    shutil.move('pgplot.ps', outname)
    shutil.move('out', 'out2d')


def call_mkimage(bindir, ra, dec, starobs):
    """
    Call mkimage, equivalent of rmkim

    Reads the out2d file and creates three images of the
    emission line data, best fit model and residuals, called
    im[123].fits.

    Parameters
    ----------
    bindir : str
        The path to the mkimage binary
    ra : float
        Right Ascension of the star.
    dec : float
        Declination of the star.
    starobs : list
        List of StarObservation objects for the star
    """

    gausa = np.loadtxt('out2d', ndmin=1, usecols=[9])

    # First write the first j4 input file
    with open('j4', 'w') as f:
        for obs in starobs:
            f.write('%f %f %f\n' % (3600.*(obs.ra-ra)
                                    * np.cos(dec/57.3),
                                    3600*(obs.dec-dec), obs.avg))

    run_command(bindir + '/mkimage')

    shutil.move('image.fits', 'im1.fits')

    with open('j4', 'w') as f:
        for i in range(0, len(starobs)):
            f.write('%f %f %f\n' % (3600.*(starobs[i].ra-ra)
                                    * np.cos(dec/57.3),
                                    3600*(starobs[i].dec-dec),
                    starobs[i].avg - gausa[i]))

    run_command(bindir + '/mkimage')

    shutil.move('image.fits', 'im2.fits')

    with open('j4', 'w') as f:
        for i in range(0, len(starobs)):
            f.write('%f %f %f\n' % (3600.*(starobs[i].ra-ra)
                                    * np.cos(dec/57.3),
                                    3600*(starobs[i].dec-dec),
                    gausa[i]))

    run_command(bindir + '/mkimage')

    shutil.move('image.fits', 'im3.fits')


def call_fitem(bindir, wl):
    """
    Call fitem requires input files created by run_fitem

    The line fitter. It fits a gauss-hermite. input is fitghsp.in.

    Parameters
    ----------
    bindir : str
        The path to the fitem binary
    wl : float
        Wavelength
    """

    input = '{wl:f}\n/vcps\n'

    run_command(bindir + '/fitem', input.format(wl=wl))


def call_sumspec(bindir, starname):
    """
    Call sumpspec. Sums a set of spectra, and then bins to 100AA bins.
    Used for SED fitting.

    Parameters
    ----------
    bindir : str
        The path to the sumspec binary.
    starname : str
        Star name used to create the outputn filename (adds specf.dat)
    """
    with open('list', 'w') as f:
        f.write(starname + 'specf.dat')

    run_command(bindir + '/sumspec')


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


def apply_factor_spline(factor):
    """
    Equivalent of the rawksp[12] scripts

    Apply the factor to the splines.out file. The factor is the number
    of individual shots the star was observed in.

    Parameters
    ----------
    factor : int
        The factor to apply.
    """
    wave, flx = np.loadtxt('splines.out', unpack=True, usecols=[0, 2])

    with open('fitghsp.in', 'w') as f:
        for w, fl in zip(wave, flx):
            f.write('%f %f\n' % (w, fl*1.e17 / factor))


def get_star_spectrum_data(ra, dec, args):
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

    if not args.multi_shot:  # rsp1b mode
        w = (night == args.night) & (shot == args.shotid)
        night = night[w]
        shot = shot[w]

    night_shots = []
    starobs = []
    c = 0

    _logger.info('Found %d shots' % len(shot))

    for n, s in zip(night, shot):
        dithall_file = args.dithall_dir+'/'+n+'v'+s+'/dithall.use'
        _logger.info('Reading dithall file %s' % dithall_file)

        try:
            ra_ifu, dec_ifu, x_ifu, y_ifu = np.loadtxt(dithall_file,
                                                       unpack=True,
                                                       usecols=[0, 1, 3, 4])
            fname_ifu, shotname_ifu, expname_ifu = \
                np.loadtxt(dithall_file, unpack=True,
                           dtype='U50', usecols=[7, 8, 9])
        except Exception as e:
            _logger.warn('Failed to read %s' % dithall_file)
            _logger.exception(e)
            continue

        w = np.where(((np.sqrt((np.cos(dec/57.3)*(ra_ifu-ra))**2
                               + (dec_ifu-dec)**2)*3600.)
                      < args.ifu_search_radius))[0]

        _logger.info('Found %d fibers' % len(w))

        for i in w:

            so = StarObservation()

            so.num = c+101
            so.night = n
            so.shot = s
            so.ra = ra_ifu[i]
            so.dec = dec_ifu[i]
            so.x = x_ifu[i]
            so.y = y_ifu[i]
            so.set_fname(fname_ifu[i])
            so.shotname = shotname_ifu[i]
            so.expname = expname_ifu[i]

            so.dist = 3600.*np.sqrt((np.cos(dec/57.3)*(so.ra-ra))**2
                                    + (so.dec-dec)**2)

            # This is written to loffset
            so.offsets_ra = 3600.*(ra_ifu[i]-ra)
            so.offsets_dec = 3600.*(dec_ifu[i]-dec)

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


def extract_star_spectrum(starobs, args, prefix=''):
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
        call_imextsp(args.bin_dir, fpath, s.ifuslot, args.extraction_wl,
                     args.extraction_wlrange,
                     get_throughput_file(args.tp_dir, s.night+'v'+s.shot),
                     args.norm_dir+'/'+s.fname+".norm",
                     prefix+'tmp%d.dat' % s.num)

        specfiles.append(prefix+'tmp%d.dat' % s.num)

    return specfiles


def get_shuffle_stars(shuffledir, nightshot, maglim):
    """
    Find the all stars for a given night / shot.

    Parameters
    ----------
    shuffledir : str
        Path to a directory where a nightshot directory with a
        shout.ifustars file.
    nightshot : str
        Night + shot name to work on.
    maglim : float
        Magnitude limit to apply to the star selection.
    """

    stars = []

    c = 1
    try:
        with _masterLock:
            indata = np.loadtxt(shuffledir + '/' + nightshot
                                + '/shout.ifustars')
        for d in indata:
            star = ShuffleStar(20000 + c, d[0], d[1], d[2], d[3], d[4], d[5],
                               d[6], d[7], d[8])
            if star.mag_g < maglim:
                stars.append(star)
                c += 1

        return stars
    except OSError:
        _logger.error('Failed to find shuffle stars for night %s, shot %s'
                      % (args.night, args.shotid))


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


def average_spectra(specfiles, starobs, wl, wlrange):
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

    with open('spavg.all', 'w') as f:
        for spf, obs in zip(specfiles, starobs):
            sp = Spectrum()
            sp.read(spf)
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

    for obs in starobs:
        fpath = '%s/%s/virus/virus%07d/%s/virus/%s' \
            % (path, obs.night, int(obs.shot),
               obs.expname, obs.fname) + '.fits'
        with fits.open(fpath, 'readonly') as hdu:
            obs.structaz = hdu[0].header['STRUCTAZ']


def run_fit2d(bindir, ra, dec, starobs, seeing, outname):
    """
    Prepare input files for running fit2d, and run it.

    Parameters
    ----------
    bindir : str
        The path to the fit2d binary
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
    with open('in', 'w') as f:
        for obs in starobs:
            f.write('%f %f %f %f %s %s %s %s %f %f\n'
                    % (obs.ra, obs.dec, obs.avg, obs.avg_norm, obs.shotname,
                       obs.night, obs.shot, obs.expname, obs.structaz,
                       obs.avg_error))
    with open('fwhm.use', 'w') as f:
        f.write('%f\n' % seeing)

    call_fit2d(bindir, ra, dec, outname)


def run_sumlineserr(bindir, specfiles):
    """
    Prepare input and run sumlineserr. It sums a set of spectra, and then bins
    to 100AA bins. Used for SED fitting.

    Parameters
    ----------
    bindir : str
        The path to the sumlineserr binary
    specfiles : list
        List of spectrum filenames.

    """

    indata = np.loadtxt('out2d', dtype='U50', ndmin=2,
                        usecols=[8, 9, 10, 11, 12, 13, 14])

    with open('list2', 'w') as f:
        for spf, d in zip(specfiles, indata):
            f.write('%s %s %s %s %s %s %s %s\n' %
                    (spf, d[0], d[1], d[2], d[3], d[4], d[5], d[6]))

    run_command(bindir + '/sumlineserr')


def run_fitem(bindir, wl, outname):
    """
    Prepare input file for fitem, and run it.

    Parameters
    ----------
    bindir : str
        The path to the sumlineserr binary
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

    indata = np.loadtxt('splines.out', dtype='U50',
                        usecols=[0, 1, 2, 3, 4])

    with open('fitghsp.in', 'w') as f:
        for d in indata:
            f.write('%s %s %s %s %s\n' %
                    (d[0], d[2], d[4], d[1], d[3]))

    call_fitem(bindir, wl)

    shutil.move('fitghsp.in', outname+'spece.dat')
    shutil.move('pgplot.ps', outname+'_2dn.ps')
    shutil.move('lines.out', outname+'_2d.res')


def run_getsdss(bindir, filename, sdss_file):
    """
    Run getsdss on filename. Equivalent to rsdss file.

    Parameters
    ----------
    bindir : str
        The path to the getsdssg binary
    filename : str
        Filename with spectral data
    sdss_file : str
        Full path and filename to the sdss g-band filter curve.

    Returns
    -------
    The flux in the g-Band.

    """
    shutil.copy2(sdss_file, 'sdssg.dat')
    shutil.copy2(filename, 's1')

    run_command(bindir + '/getsdssg')

    return float(np.loadtxt('out'))


def run_biwt(bindir, data):
    """
    Calculate biweight of the supplied data.

    Parameters
    ----------
    bindir : str
        The path to the biwt binary
    data : list
        List of the data to be run through biwt.

    Returns
    -------
    n, biwt, error
    """
    with open('tp.dat', 'w') as f:
        for d in data:
            f.write('%f\n', d)

    run_command(bindir + '/biwt', 'tp.dat\n1\n')

    return np.loadtxt('biwt.out')


def copy_stardata(starname, starid):
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

    shutil.copy2(starname+'specf.dat', 'sp%d_2.dat' % starid)
    shutil.copy2('sumspec.out', 'sp%d_100.dat' % starid)


def run_shuffle_photometry(args):
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

    stars = get_shuffle_stars(args.shuffle_ifustars_dir, nightshot,
                              args.shuffle_mag_limit)

    pool = MPPool(args.jobnum, args.shuffle_cores)

    for star in stars:

        pool.add_task(run_star_photometry, nightshot, star.ra, star.dec,
                      star.starid, copy.copy(args))

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
        stardir = curdir + '/' + starname
        if not os.path.exists(stardir):
            os.mkdir(stardir)
        os.chdir(stardir)

        # Extract data like the data in l1
        starobs, nshots = get_star_spectrum_data(ra, dec, args)

        if not len(starobs):
            _logger.warn('No shots found, skipping!')
            os.chdir(curdir)
            return

        # Call rspstar
        specfiles = extract_star_spectrum(starobs, args)

        call_sumsplines(args.bin_dir, len(starobs))

        apply_factor_spline(len(nshots))

        call_fitonevp(args.bin_dir, args.extraction_wl,
                      nightshot+'_'+str(starid))

        average_spectra(specfiles, starobs, args.extraction_wl,
                        args.average_wlrange)

        get_structaz(starobs, args.multifits_dir)

        run_fit2d(args.bin_dir, ra, dec, starobs, args.seeing,
                  starname + '.ps')

        call_mkimage(args.bin_dir, ra, dec, starobs)

        run_sumlineserr(args.bin_dir, specfiles)

        run_fitem(args.bin_dir, args.extraction_wl, starname)

        # Extract full spectrum

        fspecfiles = extract_star_spectrum(starobs, args, prefix='f')

        run_sumlineserr(args.bin_dir, fspecfiles)

        indata = np.loadtxt('splines.out', dtype='U50',
                            usecols=[0, 1, 2, 3, 4])

        with open(starname + 'specf.dat', 'w') as f:
            for d in indata:
                f.write('%s %s %s %s %s\n' % (d[0], d[2], d[4], d[1], d[3]))

        call_sumspec(args.bin_dir, starname)

        mind = args.shot_search_radius
        for o in starobs:
            if o.dist < mind:
                mind = o.dist

        _logger.info('Closest fiber is %.5f arcseconds away' % mind)

        copy_stardata(starname, starid)

        _logger.info('Saving star data for %d' % starid)
        save_data(starobs, 'sp%d.obsdata' % starid)

        # Finally save the results to the results_dir

        _logger.info('Saving data for %s' % starname)
        shutil.copy2(starname+'.ps', args.results_dir)
        shutil.copy2(starname+'_2d.res', args.results_dir)
        shutil.copy2(starname+'_2dn.ps', args.results_dir)
        shutil.copy2(starname+'spec.dat', args.results_dir)
        shutil.copy2(starname+'spec.res', args.results_dir)
        shutil.copy2(starname+'spece.dat', args.results_dir)
        shutil.copy2(starname+'specf.dat', args.results_dir)
        shutil.copy2(starname+'tot.ps', args.results_dir)
        shutil.copy2('sp%d_2.dat' % starid, args.results_dir)
        shutil.copy2('sp%d_100.dat' % starid, args.results_dir)
        shutil.copy2('sp%d.obsdata' % starid, args.results_dir)

        os.chdir(curdir)

        _logger.info('Finished star extraction for %s' % starname)
    except Exception as e:
        _logger.exception(e)


def get_g_band_throughput(args):

    nightshot = args.night + 'v' + args.shotid

    _logger.info('Reading %s.shstars in %s'
                 % (nightshot, os.path.abspath(os.path.curdir)))

    stars = read_data('%s.shstars' % nightshot)

    flxdata = []

    for s in stars:
        if not os.path.exists('sp%d.obsdata' % s.starid):
            _logger.warn('No spectral data for %d found!' % s.starid)
            continue
        starobs = read_data('sp%d.obsdata' % s.starid)
        g_flx = run_getsdss(args.bin_dir, 'sp%d_100.dat' % s.starid,
                            args.sdss_filter_file)
        sdss_flx = 5.048e-9*(10**(-0.4*s.mag_g))/(6.626e-27) / \
            (3.e18/4680.)*360.*5.e5*100

        dflx = 0.
        if sdss_flx > 0.:
            dflx = g_flx / sdss_flx

        if len(starobs) > 15 and dflx > 0.02 and dflx < 0.21:
            flxdata.append(dflx)

    avg_flx = run_biwt(args.bin_dir, flxdata)

    return avg_flx


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
            os.chdir(wdir)

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
                os.chdir(wdir)
                # Equivalent of rsetstar
                _logger.info('Extracting all shuffle stars')
                run_shuffle_photometry(args)
                _logger.info('Finished star extraction')
            if task in ['get_g_band_throughput', 'all']:
                os.chdir(wdir)
                _logger.info('Getting g-band photometry')
                get_g_band_throughput(args)

            if task in ['mk_sed_throughput_curve', 'all']:
                pass

            if task in ['fit_throughput_curve', 'all']:
                pass
    except Exception as e:
        _logger.exception(e)

    finally:
        os.chdir(args.curdir)
        vdrp_info.save(wdir)
        _logger.info("Done.")


if __name__ == "__main__":
    argv = None
    if argv is None:
        argv = sys.argv

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

    # Setup the logging system
    # logDict = {'version': 1,
    #            'formatters': {
    #                'simple': {
    #                    'format': '%(asctime)s %(levelname)-8s '
    #                    '%(threadname)-12s %(funcName)15s(): %(message)s',
    #                    'datefmt': '%m-%d %H:%M:%S'}},
    #            'handlers': {
    #                'console': {
    #                    'class': 'logging.StreamHandler',
    #                    'level': 'INFO',
    #                    'formatter': 'simple',
    #                    'stream': 'ext://sys.stdout'},
    #                'mplog': {'class': 'mplog.MultiProcessingLog',
    #                          'formatter': 'simple',
    #                          'level': 'INFO',
    #                          'maxsize': 1024,
    #                          'mode': 'w',
    #                          'name': args.logfile,
    #                          'rotate': 0}},
    #           'root': {'handlers': ['console', 'mplog'], 'level': 'DEBUG'}}

    # logging.config.dictConfig(logDict)

    fmt = '%(asctime)s %(levelname)-8s %(threadname)12s %(funcName)15s(): ' \
        '%(message)s',
    formatter = logging.Formatter(fmt, datefmt='%m-%d %H:%M:%S')
    _logger.setLevel = logging.DEBUG
    # _logger.setFormatter(formatter)

    cHndlr = logging.StreamHandler()
    cHndlr.setLevel(logging.INFO)
    cHndlr.setFormatter(formatter)

    _logger.addHandler(cHndlr)

    fHndlr = logging.FileHandler(args.logfile, mode='w')
    fHndlr.setLevel(logging.INFO)
    fHndlr.setFormatter(formatter)

    _logger.addHandler(fHndlr)

    mplog.install_mp_handler(_logger)

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

        if args.multi.find('[') != -1:
            try:
                minl, maxl = args.multi.split('[')[1].split(']')[0].split(':')
            except ValueError:
                raise Exception('Failed to parse line range, should be of '
                                'form [min:max]!')

            cmdlines = cmdlines[int(minl):int(maxl)]

        pool = ThreadPool(args.mcores)
        c = 1

        for l in cmdlines:
            largs = copy.copy(remaining_argv)
            largs += l.split()

            main_args = parseArgs(largs)

            pool.add_task(main, c, copy.copy(main_args))

        pool.wait_completion()

        sys.exit(0)
    else:
        # Parse config file and command line paramters
        # command line parameters overwrite config file.

        # The first positional argument wasn't an input list,
        # so process normally
        args = parseArgs(remaining_argv)

        sys.exit(main(args))
