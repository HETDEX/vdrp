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

# from distutils import dir_util

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


class NoShotsException(Exception):
    pass


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

        # l1 - 8q is args.extraction_wl

        self.avg = 0.
        self.avg_norm = 0.
        self.avg_error = 0.

        self.structaz = -1.

    def set_fname(self, fname):
        self.full_fname = fname
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
    defaults['photometry_logfile'] = 'photometry.log'

    defaults['starid'] = 1

    defaults['shuffle_stars'] = False

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
    parser.add_argument("--photometry_logfile", type=str,
                        help="Filename for log file.")

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

    # Commandline only paramters
    parser.add_argument("--multi_shot", action='store_true',
                        help="Run using all shots containing the star at the "
                        "given coordinates. Equivalent of rsp1 script")
    parser.add_argument("--shuffle_stars", action='store_true',
                        help="Run over all stars from shuffle for the given"
                        "night and shot, ignoring the ra and dec parameters")

    # positional arguments
    parser.add_argument('night', metavar='night', type=str,
                        help='Night of observation (e.g. 20180611).')
    parser.add_argument('shotid', metavar='shotid', type=str,
                        help='Shot ID (e.g. 017).')
    parser.add_argument('ra', metavar='ra', type=float,
                        help='RA of the target in decimal hours.')
    parser.add_argument('dec', metavar='dec', type=float,
                        help='Dec of the target in decimal hours degree.')

    args = parser.parse_args(remaining_argv)

    args.config_source = config_source
    # should in principle be able to do this with accumulate???
    # args.use_tmp = args.use_tmp == "True"
    # args.remove_tmp = args.remove_tmp == "True"

    return args


def run_command(cmd, input=None):
    logging.info('Running %s' % cmd)
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
    input = '"{filename:s}"\n{ifuslot} {wl} {wlw}\n"{tpavg}"\n"{norm}"\n'

    try:
        os.remove('out.sp')
    except OSError:
        pass

    try:
        os.remove('outfile')
    except OSError:
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
            f.write('tmp{c}.dat\n'.format(c=i+101))

    run_command(bindir + '/sumsplines')


def call_fitonevp(bindir, wave, outname):
    """
    Call fitonevp

    Requires fitghsp.in created by apply_factor_spline
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
    Call fit2d

    Requires input files generated by run_fit2d
    """
    input = '{ra:f} {dec:f}\n/vcps\n'

    run_command(bindir + '/fit2d', input.format(ra=ra, dec=dec))

    shutil.move('pgplot.ps', outname)
    shutil.move('out', 'out2d')


def call_mkimage(bindir, ra, dec, starobs):
    """
    Call mkimage, equivalent of rmkim
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
    """

    input = '{wl:f}\n/vcps\n'

    run_command(bindir + '/fitem', input.format(wl=wl))


def call_sumspec(bindir, starname):

    with open('list', 'w') as f:
        f.write(starname + 'specf.dat')

    run_command(bindir + '/sumspec')


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
    wave, flx = np.loadtxt('splines.out', unpack=True, usecols=[0, 2])

    with open('fitghsp.in', 'w') as f:
        for w, fl in zip(wave, flx):
            f.write('%f %f\n' % (w, fl*1.e17 / factor))


def get_star_spectrum_data(ra, dec, args):
    """
    This extracts the data about the different observations of the same star
    on different ifus.

    This is essentially the information stored in the l1 file.
    """

    # First find matching shots
    logging.info('Reading radec file %s' % args.radec_file)
    night, shot = np.loadtxt(args.radec_file, unpack=True, dtype='U50',
                             usecols=[0, 1])
    ra_shot, dec_shot = np.loadtxt(args.radec_file, unpack=True,
                                   usecols=[2, 3])

    logging.info('Searching for shots within %f arcseconds of %f %f'
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

    logging.info('Found %d shots' % len(shot))

    for n, s in zip(night, shot):
        dithall_file = args.dithall_dir+'/'+n+'v'+s+'/dithall.use'
        logging.info('Reading dithall file %s' % dithall_file)

        ra_ifu, dec_ifu, x_ifu, y_ifu = np.loadtxt(dithall_file,
                                                   unpack=True,
                                                   usecols=[0, 1, 3, 4])
        fname_ifu, shotname_ifu, expname_ifu = np.loadtxt(dithall_file,
                                                          unpack=True,
                                                          dtype='U50',
                                                          usecols=[7, 8, 9])

        w = np.where(((np.sqrt((np.cos(dec/57.3)*(ra_ifu-ra))**2
                               + (dec_ifu-dec)**2)*3600.)
                      < args.ifu_search_radius))[0]

        logging.info('Found %d fibers' % len(w))

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
                % (args.multifits_dirpath, so.night, int(so.shot),
                   so.expname, so.fname) + '.fits'

            if not os.path.exists(fpath):
                logging.warn('No fits data found for ifuslot %d in  %sv%s'
                             % (so.ifuslot, so.night, so.shot))
                continue

            starobs.append(so)
            night_shots.append('%s %s' % (n, s))

            c += 1

    return starobs, np.unique(night_shots)


def extract_star_spectrum(starobs, args, prefix=''):
    """
    Equivalent of the rextsp[1] and parts of the rsp1b scripts
    """

    specfiles = []

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

    stars = []

    c = 1
    try:
        indata = np.loadtxt(shuffledir + '/' + nightshot + '/shout.ifustars')
        for d in indata:
            star = ShuffleStar(20000 + c, d[0], d[1], d[2], d[3], d[4], d[5],
                               d[6], d[7], d[8])
            if star.mag_g < maglim:
                stars.append(star)
                c += 1

        return stars
    except OSError:
        logging.error('Failed to find shuffle stars for night %s, shot %s'
                      % (args.night, args.shotid))


def average_spectrum(spec, wlmin, wlmax):
    """
    Corresponds to ravgsp0 script. Calculate the average of the
    spectrum in the range [wlmin, wlmax]

    Parameters
    ----------
    spec : Spectrum
        Spectrum class object

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

    Parameters:
    -----------

    fname : string
    Filename to read from

    Read the STRUCTAZ parameter from the fits file ``fname``
    """

    for obs in starobs:
        fpath = '%s/%s/virus/virus%07d/%s/virus/%s' \
            % (path, obs.night, int(obs.shot),
               obs.expname, obs.fname) + '.fits'
        with fits.open(fpath, 'readonly') as hdu:
            obs.structaz = hdu[0].header['STRUCTAZ']


def run_fit2d(bindir, ra, dec, starobs, seeing, outname):
    """
    Prepare input files for running fit2d
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

    indata = np.loadtxt('out2d', dtype='U50',
                        usecols=[8, 9, 10, 11, 12, 13, 14])

    with open('list2', 'w') as f:
        for spf, d in zip(specfiles, indata):
            f.write('%s %s %s %s %s %s %s %s\n' %
                    (spf, d[0], d[1], d[2], d[3], d[4], d[5], d[6]))

    run_command(bindir + '/sumlineserr')


def run_fitem(bindir, wl, outname):

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


def run_shuffle_photometry(args):
    """
    Equivalent of the rsetstar script
    """
    nightshot = args.night + 'v' + args.shotid

    stars = get_shuffle_stars(args.shuffle_ifustars_dir, nightshot,
                              args.shuffle_mag_limit)

    for star in stars:
        try:
            run_star_photometry(star.ra, star.dec, star.starid, args)
        except NoShotsException:
            logging.info('No shots found for shuffle star at %f %f'
                         % (star.ra,  star.dec))


def run_star_photometry(ra, dec, starid, args):
    """
    Equivalent of the rsp1a2b script
    """
    nightshot = args.night + 'v' + args.shotid

    starname = '%s_%d' % (nightshot, starid)

    # Create the workdirectory for this star
    stardir = args.curdir + '/' + starname
    if not os.path.exists(stardir):
        os.mkdir(stardir)
    os.chdir(stardir)

    # Extract data like the data in l1
    starobs, nshots = get_star_spectrum_data(ra, dec, args)

    if not len(starobs):
        logging.warn('No shots found, skipping!')
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

    logging.info('Closest fiber is %.5f arcseconds away' % mind)

    cp_results(starname, starid, args.results_dir)

    os.chdir(args.curdir)


def cp_results(starname, starid, results_dir):
    """ Copies the result files from workdir results_dir as done by rspstar.

    Args
    ----
    workdir : str
        Work directory.
    results_dir : str
        Final directory for results.

    """

    if results_dir == '':
        logging.info('No results dir specified, skipping copying.')
        return

    if not os.path.exists:
        os.path.mkdir(results_dir)

    shutil.copy2(starname+'specf.dat',
                 os.path.join(results_dir, 'sp%d_2.dat' % starid))

    shutil.copy2('sumspec.out',
                 os.path.join(results_dir, 'sp%d_100.dat' % starid))


vdrp_info = None


def main(args):
    """
    Main function.
    """
    global vdrp_info

    # Create results directory for given night and shot
    cwd = os.getcwd()
    results_dir = os.path.join(cwd, args.night + 'v' + args.shotid + '_res')
    utils.createDir(results_dir)
    args.results_dir = results_dir

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

    # tasks = args.task.split(",")
    # if args.use_tmp and not tasks == ['all']:
    #    logging.error("Step-by-step execution not possile when running "
    #                  "in a tmp directory.")
    #    logging.error("   Please either call without -t or set "
    #                  "use_tmp to False.")
    #    sys.exit(1)

    # default is to work in results_dir
    wdir = results_dir

    logging.info("Configuration {}.".format(args.config_source))

    vdrp_info = VdrpInfo.read(wdir)
    vdrp_info.night = args.night
    vdrp_info.shotid = args.shotid

    args.curdir = os.path.abspath(os.path.curdir)

    try:
        os.chdir(wdir)
        if args.shuffle_stars:
            logging.info('Running over all shuffle stars')
            run_shuffle_photometry(args)
        else:
            logging.info('Running on a single RA/DEC position')
            run_star_photometry(args.ra, args.dec, args.starid, args)
        # for task in tasks:
        #     os.chdir(wdir)
        #     run_star_photometry(args)
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
