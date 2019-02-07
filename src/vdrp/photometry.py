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

import time
import json
import os
import shutil
import sys
import ConfigParser
import logging
import logging.config
import tempfile
import numpy as np

import vdrp.mplog as mplog
import vdrp.astrometry as astrom
import vdrp.programs as vp
import vdrp.star_extraction as vstar
import vdrp.extraction as vext
import vdrp.file_tools as vft

from distutils import dir_util

import vdrp.utils as utils
from vdrp.mphelpers import MPPool, mp_run
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


def getDefaults():

    defaults = {}

    defaults['shuffle_cores'] = 1
    defaults["shuffle_mag_limit"] = 20.
    defaults["shuffle_ifustars_dir"] = \
        '/work/00115/gebhardt/maverick/sci/panacea/test/shifts/'

    # Extraction paramters

    defaults['dithall_dir'] = '/work/00115/gebhardt/maverick/detect/'
    defaults['multifits_dir'] = '/work/03946/hetdex/maverick/red1/reductions/'
    defaults['tp_dir'] = '/work/00115/gebhardt/maverick/detect/tp/'
    defaults['norm_dir'] = '/work/00115/gebhardt/maverick/getampnorm/all/'

    defaults['radec_file'] = '/work/00115/gebhardt/maverick/getfib/radec.all'

    defaults['extraction_wl'] = 4505.
    defaults['extraction_wlrange'] = 1035.
    defaults['full_extraction_wl'] = 4500.
    defaults['full_extraction_wlrange'] = 1000.
    defaults['average_wl'] = 4500.
    defaults['average_wlrange'] = 10.

    defaults['ifu_search_radius'] = 4.
    defaults['shot_search_radius'] = 600.

    defaults['seeing'] = 1.5

    # Shuffle parameters
    defaults["acam_magadd"] = 5.
    defaults["wfs1_magadd"] = 5.
    defaults["wfs2_magadd"] = 5.
    defaults["fplane_txt"] = "$config/fplane.txt"
    defaults["shuffle_cfg"] = "$config/shuffle.cfg"

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

    return defaults


def get_arguments(parser):
    '''
    Add command line arguments for the photometry routines, this function
    can be called from another tool.

    Parameters
    ----------
    parser : argparse.ArgumentParser
    '''

    parser.add_argument("--dithall_dir", type=str, help="Base directory "
                        "used to find the dithall.use files")
    parser.add_argument("--multifits_dir", type=str, help="Directory "
                        "with the multi extension fits files")
    parser.add_argument("--tp_dir", type=str, help="Directory "
                        "with the throughput files")
    parser.add_argument("--norm_dir", type=str, help="Directory "
                        "with the amplifier normalization files")

    parser.add_argument("--radec_file", type=str, help="Filename of file with "
                        "RA DEC PA positions for all shots")

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

    parser.add_argument("--ifu_search_radius", type=float, help="Radius for "
                        "search for fibers near a given star.")
    parser.add_argument("--shot_search_radius", type=float, help="Radius for "
                        "search for shots near a given star.")

    parser.add_argument("--shuffle_cores", type=int,
                        help="Number of multiprocessing cores to use for"
                        "shuffle star extraction.")
    parser.add_argument("--shuffle_mag_limit", type=float,
                        help="Magnitude cutoff for selection of stars found by"
                        " shuffle")
    parser.add_argument("--shuffle_ifustars_dir", type=str, help="Directory "
                        "with the *ifustars shuffle output files")

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

    return parser


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

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h

    # Inherit options from config_paarser
    parser = AP(parents=[conf_parser])

    parser.set_defaults(**defaults)
    parser.add_argument("--logfile", type=str,
                        help="Filename for log file.")

    parser = get_arguments(parser)

    # Script specific parameters
    parser.add_argument("-t", "--task", type=str, default='all',
                        help="Task to execute.")

    # Boolean paramters
    parser.add_argument("--use_tmp", action='store_true',
                        help="Use a temporary directory. Result files will"
                        " be copied to NIGHTvSHOT/res.")
    parser.add_argument("--debug", action='store_true',
                        help="Keep temporary directories")

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

    args.fplane_txt = utils.mangle_config_pathname(args.fplane_txt)
    args.shuffle_cfg = utils.mangle_config_pathname(args.shuffle_cfg)

    return args


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


def extract_star_single_shot(ra, dec, starid, args, dithall=None):
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
        starobs, nshots = vstar.get_star_spectrum_data(ra, dec, args,
                                                      (args.night,
                                                       args.shotid),
                                                      False, dithall=dithall)

        if not len(starobs):
            _logger.warn('No shots found, skipping!')
            return

        # Call rspstar
        # Get fwhm and relative normalizations
        vp.call_getnormexp(nightshot, stardir)

        specfiles = vstar.extract_star_spectrum(starobs, args,
                                                args.extraction_wl,
                                                args.extraction_wlrange,
                                                stardir)

        vp.call_sumsplines(len(starobs), stardir)

        vstar.apply_factor_spline(len(nshots), stardir)

        vp.call_fitonevp(args.extraction_wl, nightshot+'_'+str(starid),
                         stardir)

        vstar.average_spectra(specfiles, starobs, args.average_wl,
                              args.average_wlrange, stardir)

        vext.get_structaz(starobs, args.multifits_dir)

        vstar.run_fit2d(ra, dec, starobs, args.seeing, starname + '.ps',
                        stardir)

        # Save the out2 file created by fit2d
        shutil.copy2(os.path.join(stardir, 'out2'),
                     os.path.join(stardir, 'sp%d_out2.dat') % starid)

        vp.call_mkimage(ra, dec, starobs, stardir)

        vstar.run_sumlineserr(specfiles, stardir)

        vstar.run_fitem(args.extraction_wl, starname, stardir)

        # Extract full spectrum

        fspecfiles = vstar.extract_star_spectrum(starobs, args,
                                                 args.full_extraction_wl,
                                                 args.full_extraction_wlrange,
                                                 stardir, prefix='f')

        vstar.run_sumlineserr(fspecfiles, stardir)

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

        vstar.copy_stardata(starname, starid, stardir)

        _logger.info('Saving star data for %d' % starid)
        save_data(stardir, os.path.join(stardir, 'sp%d.obsdata' % starid))

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
        shutil.copy2(os.path.join(stardir, starname+'specf.dat'),
                     os.path.join(args.results_dir, 'sp%d_2.dat' % starid))
        shutil.copy2(os.path.join(stardir, 'sumspec.out'),
                     os.path.join(args.results_dir, 'sp%d_100.dat' % starid))
        shutil.copy2(os.path.join(stardir, 'sp%d.obsdata') % starid,
                     args.results_dir)
        shutil.copy2(os.path.join(stardir, 'sp%d_out2.dat') % starid,
                     args.results_dir)

        _logger.info('Finished star extraction for %s' % starname)
    except Exception as e:
        _logger.exception(e)


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

    # Read in the dithall file for this night / shot
    dithall_file = vft.get_dithall_file(args.dithall_dir, args.night,
                                        args.shotid)

    _logger.info('Reading dithall file %s' % dithall_file)
    try:
        dithall = DithAllFile(dithall_file)

    except Exception as e:
        _logger.error('Failed to read %s' % dithall_file)
        _logger.exception(e)
        return

    # Parallelize the star extraction. Create a MPPool with
    # shuffle_cores processes

    pool = MPPool(args.jobnum, args.shuffle_cores)

    for star in stars:

        # Add all the tasks, they will start right away.
        pool.add_task(extract_star_single_shot(star.ra, star.dec,
                                               star.starid, args,
                                               dithall=dithall))

    # Now wait for all tasks to finish.
    pool.wait_completion()

    _logger.info('Saving star data for %s' % nightshot)

    save_data(stars, os.path.join(args.results_dir, '%s.shstars' % nightshot))


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
    # with open(os.path.join(results_dir, 'args.pickle'), 'wb') as f:
    #     pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)
    argfile = '%sv%s_%f.args.json' % (args.night, args.shotid, time.time())
    with open(os.path.join(results_dir, argfile), 'w') as f:
        json.dump(vars(args), f)

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
                vstar.run_star_photometry(args.target_ra, args.target_dec,
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
    parser.add_argument('-L', '--loglevel', type=str, default='INFO',
                        help='Loglevel to use.')

    args, remaining_argv = parser.parse_known_args()

    mplog.setup_mp_logging(args.logfile, args.loglevel)

    # Run (if requested) in threaded mode, this function will call sys.exit
    mp_run(main, args, remaining_argv, parseArgs)


if __name__ == "__main__":
    run()
