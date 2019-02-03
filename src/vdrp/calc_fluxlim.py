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
import sys
import ConfigParser
import logging
import logging.config
import copy
from astropy.io import fits
import shutil
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle

from distutils import dir_util

import vdrp.mplog as mplog
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

    defaults["remove_tmp"] = True
    defaults["debug"] = False

    defaults['fluxlim_logfile'] = 'fluxlim.log'

    defaults['dithall_dir'] = '/work/00115/gebhardt/maverick/detect/'
    defaults['multifits_dir'] = '/work/03946/hetdex/maverick/red1/reductions/'
    defaults['tp_dir'] = '/work/00115/gebhardt/maverick/detect/tp/'
    defaults['norm_dir'] = '/work/00115/gebhardt/maverick/getampnorm/all/'

    defaults['ra_range'] = 70
    defaults['dec_range'] = 70
    defaults['radec_file'] = '/work/00115/gebhardt/maverick/getfib/radec.all'
    defaults['ifu_search_radius'] = 2.5
    defaults['shot_search_radius'] = 600.
    defaults['extraction_wl'] = 4505.
    defaults['extraction_wlrange'] = 1035.
    defaults['fitradec_step'] = 0
    defaults['fitradec_nsteps'] = 1
    defaults['fitradec_w_center'] = 4505.
    defaults['fitradec_w_range'] = 3.
    defaults['fitradec_ifit1'] = 1

    defaults['fill'] = 3.
    defaults['sn'] = 8.

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

        bool_flags = ['remove_tmp']
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
    parser.add_argument("--tp_dir", type=str, help="Directory "
                        "with the throughput files")
    parser.add_argument("--norm_dir", type=str, help="Directory "
                        "with the amplifier normalization files")

    parser.add_argument("--ra_range", type=int, help="Width in RA"
                        " direction for search grid in asec")
    parser.add_argument("--dec_range", type=int, help="Width in DEC"
                        " direction for search grid in asec")
    parser.add_argument("--radec_file", type=str, help="Filename of file with "
                        "RA DEC PA positions for all shots")
    parser.add_argument("--ifu_search_radius", type=float, help="Radius for "
                        "search for fibers near a given star.")
    parser.add_argument("--shot_search_radius", type=float, help="Radius for "
                        "search for shots near a given star.")
    parser.add_argument("--extraction_wl", type=float, help="Central "
                        "wavelength for the extraction")
    parser.add_argument("--extraction_wlrange", type=float, help="Wavelength "
                        "range for the extraction")

    parser.add_argument("--fitradec_step", type=float, help="Starting step"
                        " for fitradecsp call")
    parser.add_argument("--fitradec_nsteps", type=float, help="Number of steps"
                        " for fitradecsp call")
    parser.add_argument("--fitradec_w_center", type=float, help="Center wl"
                        " for fitradecsp call")
    parser.add_argument("--fitradec_w_range", type=float, help="Wavelength"
                        " range for fitradecsp call")
    parser.add_argument("--fitradec_ifit1", type=float, help="fit flag"
                        " for fitradecsp call")
    parser.add_argument("--fill", type=float, help="Fill value")
    parser.add_argument("--sn", type=float, help="SNR value")

    # Boolean paramters

    # positional arguments
    parser.add_argument('ra', metavar='ra', type=float,
                        help='Right Ascension.')
    parser.add_argument('dec', metavar='dec', type=float,
                        help='Declination.')
    parser.add_argument('night', metavar='night', type=str,
                        help='Night of observation (e.g. 20180611).')
    parser.add_argument('shotid', metavar='shotid', type=str,
                        help='Shotname of observation (e.g. 021).')
    parser.add_argument('fname', metavar='fname', type=str,
                        help='Basename of the multifits file.')

    args = parser.parse_args(remaining_argv)

    args.config_source = config_source
    # should in principle be able to do this with accumulate???
    # args.use_tmp = args.use_tmp == "True"
    # args.remove_tmp = args.remove_tmp == "True"

    # args.fplane_txt = utils.mangle_config_pathname(args.fplane_txt)
    # args.shuffle_cfg = utils.mangle_config_pathname(args.shuffle_cfg)

    args.nightshot = '%sv%s' % (args.night, args.shotid)

    return args


def calc_fluxlim(args):
    """
    Equivalent of the rflim0 script and of mklistfl and the rspfl3f scripts.

    Calculate the flux limit for a given night and shot, looping over a
    (by default) 70 x 70 arcsecond grid

    Paramters
    ---------
    args : struct
        The arguments structure
    """

    curdir = os.path.abspath(os.path.curdir)
    cosd = np.cos(args.dec / 57.3)
    rstart = args.ra - 35./3600./cosd
    dstart = args.dec - 35./3600.

    wave_min = args.extraction_wl - args.extraction_wlrange
    wave_max = args.extraction_wl + args.extraction_wlrange
    n_wave = int((wave_max - wave_min) / 2.) + 1

    allspec = np.full((n_wave, 70*70, 4), -9999.)

    dithall_file = args.dithall_dir+'/'+args.night + 'v' \
        + args.shotid+'/dithall.use'

    _logger.info('Reading dithall file %s' % dithall_file)
    try:
        dithall = DithAllFile(dithall_file)

    except Exception as e:
        _logger.warn('Failed to read %s' % dithall_file)
        _logger.exception(e)
        return

    counter = 0
    # This counter tracks the number of extracted spectra
    speccounter = 0

    cosdec = np.cos(args.dec/57.3)

    for r_off in range(0, args.ra_range):
        ra = rstart + r_off/3600./cosd
        for d_off in range(0, args.dec_range):
            dec = dstart + d_off/3600.
            counter += 1

            _logger.info('Working at #%d %f %f' % (counter, ra, dec))

            wdir = curdir + '/%s_%d' % (args.nightshot, counter)
            _logger.info('Creating workdir %s' % wdir)
            if not os.path.exists(wdir):
                os.mkdir(wdir)
            os.chdir(wdir)

            try:
                starobs, _ = phot.get_star_spectrum_data(ra, dec, args,
                                                         False, dithall)

                if not len(starobs):
                    raise Exception('No shots found, skipping!')

                # Call rspstar
                # Get fwhm and relative normalizations
                vp.call_getnormexp(args.nightshot)

                specfiles = phot.extract_star_spectrum(starobs, args,
                                                       args.extraction_wl,
                                                       args.extraction_wlrange)

                phot.get_structaz(starobs, args.multifits_dir)

                vp.run_fitradecsp(ra, dec, args.fitradec_step,
                                  args.fitradec_nsteps, args.fitradec_w_center,
                                  args.fitradec_w_range, args.fitradec_ifit1,
                                  starobs, specfiles)

                # Now produce the final output

                if not os.path.exists('spec.out'):
                    raise Exception('fitradecsp failed!')

            except Exception as e:
                _logger.error(e.message)
                os.chdir(curdir)
                if not args.debug:
                    shutil.rmtree(wdir)
                continue

            specdata = np.loadtxt('spec.out')

            w = np.where(specdata[:, 8] > 0)[0]

            allspec[:, speccounter, 0] = np.full_like(allspec[:, 0, 0],
                                                      3600. * (ra-args.ra)
                                                      * cosdec)
            allspec[:, speccounter, 1] = np.full_like(allspec[:, 0, 0],
                                                      3600. * (dec-args.dec))
            allspec[:, speccounter, 2] = np.full_like(allspec[:, 0, 0], -9999.)
            allspec[:, speccounter, 3] = np.full_like(allspec[:, 0, 0], -9999.)
            allspec[w, speccounter, 2] = specdata[w, 2]*specdata[w, 7] \
                / (specdata[w, 8] * args.fill) * args.sn
            allspec[w, speccounter, 3] = specdata[w, 4]*specdata[w, 7] \
                / (specdata[w, 8] * args.fill) * args.sn

            speccounter += 1

            os.chdir(curdir)
            if not args.debug:
                shutil.rmtree(wdir)

    # Now write all the spec files and the list.
    os.chdir(curdir)

    with open('list', 'w') as f:
        for i in range(0, n_wave):
            wl = int(wave_min + i*2.)

            w = np.where(allspec[i, :, 2] > -9000.)
            np.savetxt('w%d.j4' % wl, allspec[i, w[0], :],
                       fmt="%.5f %.5f %.3f %.3f")

            f.write('%s' % 'w%d.j4\n' % wl)

    vp.call_mkimage3d()

    update_im3d_header(args.ra, args.dec)

    outname = args.results_dir + '/' + args.nightshot + '_' \
        + args.fname + '.fits'
    if os.path.exists(outname):
        os.remove(outname)
    shutil.move('image3d.fits', outname)


def update_im3d_header(ra, dec):
    """
    Add header keywords to the image3d.fits
    """
    with fits.open('image3d.fits', 'update') as hdu:

        hdu[0].header['OBJECT'] = 'CAT'
        hdu[0].header['CRVAL1'] = ra
        hdu[0].header['CRVAL2'] = dec
        hdu[0].header['CRVAL3'] = 3470.0
        hdu[0].header['CDELT3'] = 2.0
        hdu[0].header['CTYPE1'] = 'RA---TAN'
        hdu[0].header['CTYPE2'] = 'DEC--TAN'
        hdu[0].header['CTYPE3'] = 'Wave'
        hdu[0].header['CD1_1'] = 0.0002777
        hdu[0].header['CD1_2'] = 0
        hdu[0].header['CD2_2'] = 0.0002777
        hdu[0].header['CD2_1'] = 0
        hdu[0].header['CRPIX1'] = 35.0
        hdu[0].header['CRPIX2'] = 35.0
        hdu[0].header['CRPIX3'] = 1
        hdu[0].header['CUNIT1'] = 'deg'
        hdu[0].header['CUNIT2'] = 'deg'
        hdu[0].header['EQUINOX'] = 2000


vdrp_info = None


def main(jobnum, args):
    """
    Main function.
    """
    global vdrp_info

    # Create results directory for given night and shot
    cwd = _baseDir
    results_dir = cwd
    utils.createDir(results_dir)
    args.results_dir = results_dir

    # save arguments for the execution
    with open(os.path.join(results_dir, 'args.pickle'), 'wb') as f:
        pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)

    # _logger.info("Executing task : {}".format(task))

    # default is to work in results_dir
    # Create a temporary directory
    tmp_dir = os.path.join(cwd, args.nightshot + '_' + args.fname)
    _logger.info("Tempdir is {}".format(tmp_dir))
    _logger.info("Copying over prior data (if any)...")
    # dir_util.copy_tree(results_dir, tmp_dir)
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

        _logger.info('Starting flux limit calculation')
        calc_fluxlim(args)
        _logger.info('Finished flux limit calculation')
    except Exception as e:
        _logger.exception(e)

    finally:
        os.chdir(args.curdir)
        vdrp_info.save(wdir)
        if not args.debug:
            shutil.rmtree(wdir)
        _logger.info("Done.")


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

    mplog.setup_mp_logging(args.logfile)

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
    calc_fluxlim_entrypoint()
