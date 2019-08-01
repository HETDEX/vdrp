#!/usr/bin/env python
""" Star Extraction routine. Equivalent of rsp1 script.

Extract star at a given RA/DEC using all shots overlapping with
these coordinates.

Contains python translation of Karl Gebhardt

.. moduleauthor:: Jan Snigula <snigula@mpe.mpg.de>
"""

from __future__ import print_function

from argparse import RawDescriptionHelpFormatter as ap_RDHF
from argparse import ArgumentParser as AP

import time
import os
import sys
import ConfigParser
import logging
import logging.config
import tempfile
import shutil
import numpy as np
import json

import vdrp.mplog as mplog
import vdrp.utils as utils
import vdrp.programs as vp
import vdrp.extraction as vext

from vdrp.mphelpers import mp_run
from vdrp.vdrp_helpers import save_data, run_command
import vdrp.containers as vcont


_baseDir = os.getcwd()

_logger = logging.getLogger()


def getDefaults():

    defaults = {}

    defaults['tmp_dir'] = '/tmp/'

    defaults['dithall_dir'] = '/work/00115/gebhardt/maverick/detect/'
    defaults['multifits_dir'] = '/work/03946/hetdex/maverick/red1/reductions/'
    defaults['tp_dir'] = '/work/00115/gebhardt/maverick/detect/tp/'
    defaults['norm_dir'] = '/work/00115/gebhardt/maverick/getampnorm/all/'

    defaults['radec_file'] = '/work/00115/gebhardt/maverick/getfib/radec.all'

    defaults['extraction_wl'] = 4505.
    defaults['extraction_wlrange'] = 1035.

    defaults['ifu_search_radius'] = 4.
    defaults['shot_search_radius'] = 600.

    defaults['extraction_wl'] = 4505.
    defaults['extraction_wlrange'] = 1035.
    defaults['full_extraction_wl'] = 4500.
    defaults['full_extraction_wlrange'] = 1000.
    defaults['average_wl'] = 4500.
    defaults['average_wlrange'] = 10.

    defaults['seeing'] = 1.5

    return defaults


def get_arguments(parser):
    '''
    Add command line arguments for the photometry routines, this function
    can be called from another tool.

    Parameters
    ----------
    parser : argparse.ArgumentParser
    '''

    parser.add_argument("--tmp_dir", type=str, help="Base directory "
                        "used to create the temporary work directory")

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
        defaults.update(dict(config.items("StarExtract")))

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h

    # Inherit options from config_paarser
    parser = AP(parents=[conf_parser])

    parser.set_defaults(**defaults)
    parser.add_argument("--logfile", type=str,
                        help="Filename for log file.")

    parser = get_arguments(parser)

    # Script specific parameters
    parser.add_argument("-t", "--task", type=str, help="Task to execute.")

    # Boolean paramters
    parser.add_argument("--use_tmp", action='store_true',
                        help="Use a temporary directory. Result files will"
                        " be copied to NIGHTvSHOT/res.")
    parser.add_argument("--debug", action='store_true',
                        help="Keep temporary directories")

    # positional arguments
    parser.add_argument('ra', metavar='ra', type=float,
                        help='Right Ascension of star.')
    parser.add_argument('dec', metavar='dec', type=float,
                        help='Declination of star.')
    parser.add_argument('starid', metavar='starid', type=int,
                        help='Star ID')

    args = parser.parse_args(remaining_argv)

    args.config_source = config_source
    # should in principle be able to do this with accumulate???
    # args.use_tmp = args.use_tmp == "True"
    # args.remove_tmp = args.remove_tmp == "True"

    return args


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
            sp = vcont.Spectrum()
            sp.read(os.path.join(wdir, spf))
            obs.avg, obs.avg_norm, obs.avg_error = \
                average_spectrum(sp, wlmin, wlmax)

            f.write('%f %.7f %.4f\n' % (obs.avg, obs.avg_norm, obs.avg_error))


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

    shutil.move(os.path.join(wdir, 'fitghsp.in'),
                os.path.join(wdir, outname+'spece.dat'))
    shutil.move(os.path.join(wdir, 'pgplot.ps'),
                os.path.join(wdir, outname+'_2dn.ps'))
    shutil.move(os.path.join(wdir, 'lines.out'),
                os.path.join(wdir, outname+'_2d.res'))


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


def extract_star(args):
    """
    Equivalent of the rsp1a2b script.

    Run the stellar extraction code for a given ra / dec position.

    Parameters
    ----------
    args : struct
        The arguments structure

    """
    try:
        _logger.info('Starting star extraction')

        starname = '%f_%f_%d' % (args.ra, args.dec, args.starid)

        _logger.info('Extracting star %s' % starname)

        # Create the workdirectory for this star
        # curdir = os.path.abspath(os.path.curdir)
        curdir = args.wdir
        stardir = os.path.join(curdir, starname)
        if not os.path.exists(stardir):
            os.mkdir(stardir)

        # Extract data like the data in l1
        starobs, nshots = \
            vext.get_star_spectrum_data(args.ra, args.dec, args,
                                        None, True)

        if not len(starobs):
            _logger.warn('No shots found, skipping!')
            return

        specfiles = vext.extract_star_spectrum(starobs, args,
                                               args.extraction_wl,
                                               args.extraction_wlrange,
                                               stardir)

        vp.call_sumsplines(len(starobs), stardir)

        apply_factor_spline(len(nshots), stardir)

        _logger.info('Finished star extraction for %s' % starname)
    except Exception as e:
        _logger.exception(e)


def main(jobnum, args):
    """
    Main function.
    """
    global vdrp_info

    # Create results directory for given night and shot
    cwd = _baseDir
    results_dir = os.path.join(cwd, '%f_%f_%d'
                               % (args.ra, args.dec, args.starid),  'res')
    utils.createDir(results_dir)
    args.results_dir = results_dir

    # save arguments for the execution
    # with open(os.path.join(results_dir, 'args.pickle'), 'wb') as f:
    #     pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)
    argfile = '%f_%f_%f.args.json' % (args.ra, args.dec, time.time())
    with open(os.path.join(results_dir, argfile), 'w') as f:
        json.dump(vars(args), f)

    # default is to work in results_dir
    wdir = results_dir
    if args.use_tmp:
        # Create a temporary directory
        tmp_dir = tempfile.mkdtemp(dir=args.tmp_dir)
        _logger.info("Tempdir is {}".format(tmp_dir))
        # _logger.info("Copying over prior data (if any)...")
        # dir_util.copy_tree(results_dir, tmp_dir)
        # set working directory to tmp_dir
        wdir = tmp_dir

    _logger.info("Configuration {}.".format(args.config_source))

    # vdrp_info = VdrpInfo.read(wdir)
    # vdrp_info.night = args.night
    # vdrp_info.shotid = args.shotid

    args.curdir = os.path.abspath(os.path.curdir)
    args.wdir = wdir
    args.jobnum = jobnum

    try:
        # Equivalent of rsp1
        _logger.info('Extracting specific RA/DEC position')
        extract_star(args)
    except Exception as e:
        _logger.exception(e)

    finally:
        # os.chdir(args.curdir)
        # vdrp_info.save(wdir)
        _logger.info("Done.")


def star_extract_entrypoint():

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

    # Setup the logging
    mplog.setup_mp_logging(args.logfile, args.loglevel)

    # Run (if requested) in threaded mode, this function will call sys.exit
    mp_run(main, args, remaining_argv, parseArgs)


if __name__ == "__main__":
    star_extract_entrypoint()
