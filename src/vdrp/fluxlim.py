#!/usr/bin/env python
""" Fluxlimit routine

Contains python translation of Karl Gebhardt

.. moduleauthor:: Jan Snigula <snigula@mpe.mpg.de>
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
import logging.config
import copy
import subprocess
from astropy.io import fits
import astropy.stats as aps
# from astropy.io import ascii
import tempfile
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle

import vdrp.mplog as mplog
import vdrp.astrometry as astrom

from distutils import dir_util

import vdrp.utils as utils
import vdrp.photometry as phot
import vdrp.programs as vp

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


def getDefaults():

    defaults = {}

    defaults["use_tmp"] = False
    defaults["remove_tmp"] = True

    defaults['fluxlim_logfile'] = 'fluxlim.log'

    defaults['dithall_dir'] = '/work/00115/gebhardt/maverick/detect/'
    defaults['multifits_dir'] = '/work/03946/hetdex/maverick/red1/reductions/'

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
        defaults.update(dict(config.items("FluxLim")))

        bool_flags = ['use_tmp', 'remove_tmp']
        for bf in bool_flags:
            if config.has_option('FluxLim', bf):
                defaults[bf] = config.getboolean('FluxLim', bf)

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h

    # Inherit options from config_paarser
    parser = AP(parents=[conf_parser])

    parser.set_defaults(**defaults)
    parser.add_argument("--fluxlim_logfile", type=str,
                        help="Filename for log file.")

    parser.add_argument("--dithall_dir", type=str, help="Base directory "
                        "used to find the dithall.use files")
    parser.add_argument("--multifits_dir", type=str, help="Directory "
                        "with the multi extension fits files")

    # Boolean paramters
    parser.add_argument("--use_tmp", action='store_true',
                        help="Use a temporary directory. Result files will"
                        " be copied to NIGHTvSHOT/res.")

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

    # args.fplane_txt = utils.mangle_config_pathname(args.fplane_txt)
    # args.shuffle_cfg = utils.mangle_config_pathname(args.shuffle_cfg)

    return args


def setup_fluxlim(args):
    """
    This is equivalent to the rflim0 and rsetfl scripts.

    Determine the input values for the flux limit calculation,
    create the input file, create the slurm file using the jobsplitter
    and launch it using sbatch
    """

    nightshot = args.night+'v'+args.shotid
    dithall = DithAllFile(args.dithall_dir+'/'+nightshot +
                          '/dithall.use')

    ifus = np.unique(dithall.ifuslot)

    with open('flim%s' % nightshot, 'w') as f:

        for ifu in ifus:
            ifu_dith = dithall.where(dithall.ifuslot == ifu)
            dist = np.sqrt(ifu_dith.x*ifu_dith.x + ifu_dith.y*ifu_dith.y)
            sortidx = np.argsort(dist)

            ra_mean = aps.biweight_location(ifu_dith.ra[sortidx][0:2])
            dec_mean = aps.biweight_location(ifu_dith.dec[sortidx][0:2])

            fname = ifu_dith.filename[sortidx][0]

            f.write('vdrp_calc_flim %.7f %f %s %s\n'
                    % (ra_mean, dec_mean, nightshot,
                       '_'.join(fname.split('_')[0:4])))


def calc_fluxlim(args):
    """
    Equivalent of the rflim0 script.

    Calculate the flux limit for a given night and shot.

    Paramters
    ---------
    args : struct
        The arguments structure
    """

    # Simulate the combination of mklistfl, and executing its output file

    pass


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


def extract_star_spectrum(starobs, args, wl, wlr, prefix=''):
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
        call_imextsp(args.bin_dir, fpath, s.ifuslot, wl, wlr,
                     get_throughput_file(args.tp_dir, s.night+'v'+s.shot),
                     args.norm_dir+'/'+s.fname+".norm",
                     prefix+'tmp%d.dat' % s.num)
        specfiles.append(prefix+'tmp%d.dat' % s.num)
    return specfiles


def get_shuffle_stars(nightshot, args):
    """
    Rerun shuffle and find the all stars for a given night / shot.

    Parameters
    ----------
    nightshot : str
        Night + shot name to work on.
    args : argparse.Namespace
        The script parameter namespace
    """

    astrom.get_ra_dec_orig('./', args.multifits_dir, args.night, args.shotid)
    track = astrom.get_track('./', args.multifits_dir, args.night, args.shotid)

    ra, dec, _ = \
        utils.read_radec("radec.orig")

    # Try to run shuffle using different catalogs, we need SDSS for SED fitting,
    # the others can be used for throughput calculation
    for cat in 'SDSS', 'GAIA', 'USNO':

        stars = []

        astrom.redo_shuffle('./', ra, dec, track,
                            args.acam_magadd, args.wfs1_magadd,
                            args.wfs2_magadd, args.shuffle_cfg,
                            args.fplane_txt, args.night, catalog=cat)

        c = 1
        try:
            indata_str = np.loadtxt('shout.ifustars', dtype='U50',
                                    usecols=[0, 1])
            indata_flt = np.loadtxt('shout.ifustars', dtype=float,
                                    usecols=[2, 3, 4, 5, 6, 7, 8])
            for ds, df in zip(indata_str, indata_flt):
                star = ShuffleStar(20000 + c, ds[0], ds[1], df[0], df[1], df[2],
                                   df[3], df[4], df[5], df[6], cat)
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


def get_sedfits(starobs, args):
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
        qf_args.filename = 'qf.ifus'
        qf_args.outfolder = './'
        qf_args.ebv = args.quick_fit_ebv
        qf_args.make_plot = args.quick_fit_plot
        qf_args.wave_init = args.quick_fit_wave_init
        qf_args.wave_final = args.quick_fit_wave_final
        qf_args.bin_size = args.quick_fit_bin_size

        have_stars = False

        with open('qf.ifus', 'w') as f:
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
                fitsedname = '%s_%s.txt' % (s.shotid, s.shuffleid)
                sedname = 'sp%d_fitsed.dat' % s.starid
                if not os.path.exists(fitsedname):
                    _logger.warn('No sed fit found for star %d' % s.starid)
                    continue
                shutil.move(fitsedname, sedname)

    except ImportError:
        _logger.warn('Failed to import quick_fit, falling back to '
                     'pre-existing SED fits')
        for s in starobs:
            fitsedname = '%s_%s.txt' % (s.shotid, s.shuffleid)
            sedname = 'sp%d_fitsed.dat' % s.starid
            if not os.path.exists(os.path.join(args.sed_fit_dir, fitsedname)):
                _logger.warn('No sed fit found for star %d' % s.starid)
                continue
            shutil.copy2(os.path.join(args.sed_fit_dir, fitsedname), sedname)


def run_fit2d(ra, dec, starobs, seeing, outname):
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
    with open('in', 'w') as f:
        for obs in starobs:
            f.write('%f %f %f %f %s %s %s %s %f %f\n'
                    % (obs.ra, obs.dec, obs.avg, obs.avg_norm, obs.shotname,
                       obs.night, obs.shot, obs.expname, obs.structaz,
                       obs.avg_error))
    if not os.path.exists('fwhm.use'):
        _logger.warn('No fwhm from getnormexp found, using default')
        with open('fwhm.use', 'w') as f:
            f.write('%f\n' % seeing)

    call_fit2d(utils._vdrp_bindir, ra, dec, outname)


def run_sumlineserr(specfiles):
    """
    Prepare input and run sumlineserr. It sums a set of spectra, and then bins
    to 100AA bins. Used for SED fitting.

    Parameters
    ----------
    specfiles : list
        List of spectrum filenames.

    """

    indata = np.loadtxt('out2d', dtype='U50', ndmin=2,
                        usecols=[8, 9, 10, 11, 12, 13, 14])

    with open('list2', 'w') as f:
        for spf, d in zip(specfiles, indata):
            f.write('%s %s %s %s %s %s %s %s\n' %
                    (spf, d[0], d[1], d[2], d[3], d[4], d[5], d[6]))

    run_command(utils._vdrp_bindir + '/sumlineserr')


def run_fitem(wl, outname):
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

    indata = np.loadtxt('splines.out', dtype='U50',
                        usecols=[0, 1, 2, 3, 4])

    with open('fitghsp.in', 'w') as f:
        for d in indata:
            f.write('%s %s %s %s %s\n' %
                    (d[0], d[2], d[4], d[1], d[3]))

    call_fitem(utils._vdrp_bindir, wl)

    shutil.move('fitghsp.in', outname+'spece.dat')
    shutil.move('pgplot.ps', outname+'_2dn.ps')
    shutil.move('lines.out', outname+'_2d.res')


def run_getsdss(filename, sdss_file):
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
    shutil.copy2(sdss_file, 'sdssg.dat')
    shutil.copy2(filename, 's1')

    run_command(utils._vdrp_bindir + '/getsdssg')

    return float(np.loadtxt('out'))


def run_biwt(data, outfile):
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
    with open('tp.dat', 'w') as f:
        for d in data:
            f.write('%f\n' % d)

    run_command(bindir + '/biwt', 'tp.dat\n1\n')

    os.remove('tp.dat')

    shutil.move('biwt.out', outfile)


def run_combsed(bindir, sedlist, sigmacut, rmscut, outfile, plotfile=None):
    """


    Parameters
    ----------
    bindir : str
        The path to the biwt binary
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
    with open('list', 'w') as f:
        for l in sedlist:
            f.write('%s\n' % l)

    input = '{:f} {:f}\n'
    print('combsed', input.format(sigmacut, rmscut))
    run_command(bindir + '/combsed', input.format(sigmacut, rmscut))

    shutil.move('comb.out', outfile)

    if plotfile is not None:

        fdata = np.loadtxt('out', dtype=float,
                           usecols=[1, 2, 4, 5])
        idata = np.loadtxt('out', dtype=int, usecols=[0, 3])

        with open('in', 'w') as f, open('in2', 'w') as f2:
            for di, df, sf in zip(idata, fdata, sedlist):
                if di[1] == 0:
                    f.write('%s %d %f %f %d %f %f\n'
                            % (sf, di[0], df[0], df[1],
                               di[1], df[2], df[3]))
                    f2.write('%s %d %f %f %d %f %f\n'
                             % (sf, di[0], df[0], df[1],
                                di[1], df[2], df[3]))
            f2.write('%s\n' % outfile)

        run_command(bindir + '/plotseda', '/vcps\n')

        shutil.move('pgplot.ps', plotfile)


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

    stars = get_shuffle_stars(nightshot, args)

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
        # Get fwhm and relative normalizations
        call_getnormexp(args.bin_dir, nightshot)

        specfiles = extract_star_spectrum(starobs, args,
                                          args.extraction_wl,
                                          args.extraction_wlrange)

        call_sumsplines(args.bin_dir, len(starobs))

        apply_factor_spline(len(nshots))

        call_fitonevp(args.bin_dir, args.extraction_wl,
                      nightshot+'_'+str(starid))

        average_spectra(specfiles, starobs, args.average_wl,
                        args.average_wlrange)

        get_structaz(starobs, args.multifits_dir)

        run_fit2d(args.bin_dir, ra, dec, starobs, args.seeing,
                  starname + '.ps')

        # Save the out2 file created by fit2d
        shutil.copy2('out2', 'sp%d_out2.dat' % starid)

        call_mkimage(args.bin_dir, ra, dec, starobs)

        run_sumlineserr(args.bin_dir, specfiles)

        run_fitem(args.bin_dir, args.extraction_wl, starname)

        # Extract full spectrum

        fspecfiles = extract_star_spectrum(starobs, args,
                                           args.full_extraction_wl,
                                           args.full_extraction_wlrange,
                                           prefix='f')

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
        shutil.copy2('sp%d_out2.dat' % starid, args.results_dir)

        os.chdir(curdir)

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

    run_biwt(args.bin_dir, flxdata, 'tp.biwt')


def mk_sed_throughput_curve(args):
    '''
    Equivalent of the rgett0b script.

    Parameters
    ----------
    args : struct
        The arguments structure

    '''

    nightshot = args.night + 'v' + args.shotid

    _logger.info('Reading %s.shstars in %s'
                 % (nightshot, os.path.abspath(os.path.curdir)))

    stars = read_data('%s.shstars' % nightshot)

    sedlist = []

    get_sedfits(stars, args)

    for s in stars:
        if not os.path.exists('sp%s_100.dat' % s.starid):
            _logger.info('No star data found for sp%s_100.dat' % s.starid)
            continue
        sedname = 'sp%d_fitsed.dat' % s.starid

        if not os.path.exists(sedname):
            _logger.warn('No sed fit found for star %d' % s.starid)
            continue

        seddata = np.loadtxt(sedname, ndmin=1).transpose()
        stardata = np.loadtxt('sp%s_100.dat' % s.starid).transpose()

        sedcgs = seddata[1][1:]/6.626e-27/(3.e18/seddata[0][1:])*360.*5.e5*100.

        np.savetxt('sp%dsed.dat' % s.starid, zip(seddata[0][1:],
                                                 seddata[1][1:]/sedcgs))

        sedlist.append('sp%dsed.dat' % s.starid)

    if not len(sedlist):
        _logger.warn('No SED fits found, skipping SED throughput curve '
                     'generation')
        return

    run_combsed(args.bin_dir, sedlist, args.sed_sigma_cut, args.sed_rms_cut,
                '%ssedtp.dat' % nightshot, '%ssedtpa.ps' % nightshot)

    data = np.loadtxt('%ssedtp.dat' % nightshot)

    with open('%sfl.dat' % nightshot, 'w') as f:
        for d in data:
            f.write('%f %f\n' % (d[0], 6.626e-27*(3.e18/d[0])/360.
                                 / 5.e5/d[1]*250.))

    with open('offsets.dat', 'w') as off:
        for star in stars:
            use_star = False
            if not os.path.exists('sp%s_100.dat' % star.starid):
                continue
            with open('sp%s_100.dat' % star.starid, 'r') as f:
                for l in f.readlines():
                    w, v = l.split()
                    if w.strip().startswith('4540') and float(v) > 10000.:
                        use_star = True
                        break
            if use_star:
                with open('sp%d_out2.dat' % star.starid, 'r') as f:
                    line = f.readline()
                    vals = line.split()
                    if float(vals[3]) > -0.5 and float(vals[3]) < 0.5 and \
                       float(vals[4]) > -0.5 and float(vals[4]) < 0.5:
                        off.write('%d %s' % (star.starid, line))


vdrp_info = None


def main(jobnum, args, task):
    """
    Main function.
    """
    global vdrp_info

    # Create results directory for given night and shot
    cwd = _baseDir
    results_dir = os.path.join(cwd, args.night + 'v' + args.shotid,  'flim')
    utils.createDir(results_dir)
    args.results_dir = results_dir

    # save arguments for the execution
    with open(os.path.join(results_dir, 'args.pickle'), 'wb') as f:
        pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)

    _logger.info("Executing task : {}".format(task))

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
        os.chdir(wdir)

        if task == 'setup_flim':
            _logger.info('Setting up flux limit calculation')
            setup_fluxlim(args)
            _logger.info('Done setting up flux limit calculation')
        elif task == 'calc_flim':
            _logger.info('Starting flux limit calculation')
            calc_fluxlim(args)
            _logger.info('Finished flux limit calculation')
        else:
            raise Exception('Unknown task %s' % task)
    except Exception as e:
        _logger.exception(e)

    finally:
        os.chdir(args.curdir)
        vdrp_info.save(wdir)
        _logger.info("Done.")


def setup_logging(logfile):
    '''
    Setup the logging and prepare it for use with multiprocessing
    '''

    # Setup the logging
    fmt = '%(asctime)s %(levelname)-8s %(threadName)12s %(funcName)15s(): ' \
        '%(message)s'
    formatter = logging.Formatter(fmt, datefmt='%m-%d %H:%M:%S')
    _logger.setLevel(logging.DEBUG)

    cHndlr = logging.StreamHandler()
    cHndlr.setLevel(logging.DEBUG)
    cHndlr.setFormatter(formatter)

    _logger.addHandler(cHndlr)

    fHndlr = logging.FileHandler(logfile, mode='w')
    fHndlr.setLevel(logging.DEBUG)
    fHndlr.setFormatter(formatter)

    _logger.addHandler(fHndlr)

    # Wrap the log handlers with the MPHandler, this is essential for the use
    # of multiprocessing, otherwise, tasks will hang.
    mplog.install_mp_handler(_logger)


def calc_fluxlim_entrypoint():

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

    setup_logging(args.logfile)

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

            pool.add_task(main, c, copy.copy(main_args), 'calc_flim')

        # Wait for all tasks to complete
        pool.wait_completion()

        sys.exit(0)
    else:
        # Parse config file and command line paramters
        # command line parameters overwrite config file.

        # The first positional argument wasn't an input list,
        # so process normally
        args = parseArgs(remaining_argv)

        sys.exit(main(1, args, 'calc_flim'))


def setup_fluxlim_entrypoint():
    '''
    Entrypoint to run the flux limit calculation for one night / shot
    combination

    '''
    # Here we create another external argument parser, this checks if we
    # are supposed to run in multi-threaded mode.

    # First check if we should loop over an input file
    parser = AP(description='Test', formatter_class=ap_RDHF, add_help=False)
    parser.add_argument('-l', '--logfile', type=str, default='vdrp.log',
                        help='Logfile to write to.')

    args, remaining_argv = parser.parse_known_args()

    setup_logging(args.logfile)

    args = parseArgs(remaining_argv)

    sys.exit(main(1, args, 'setup_flim'))


if __name__ == "__main__":
    setup_fluxlim_entrypoint()
