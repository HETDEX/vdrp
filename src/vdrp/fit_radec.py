#!/usr/bin/env python
""" RA /DEC fitting routine, equivalent of rsp3f script

Contains python translation of Karl Gebhardt

.. moduleauthor:: Jan Snigula <snigula@mpe.mpg.de>
"""

from __future__ import print_function
# import matplotlib

# from matplotlib import pyplot as plt

from argparse import RawDescriptionHelpFormatter as ap_RDHF
from argparse import ArgumentParser as AP

import os
import ConfigParser
import logging
import logging.config
import shutil

import vdrp.mplog as mplog
import vdrp.programs as vp
import vdrp.extraction as vext
# import vdrp.star_extraction as vstar

from vdrp.mphelpers import mp_run
# from vdrp.vdrp_helpers import VdrpInfo, save_data, read_data, run_command
from vdrp.containers import DithAllFile


_baseDir = os.getcwd()

_logger = logging.getLogger()


def getDefaults():

    defaults = {}

    defaults['dithall_dir'] = '/work/00115/gebhardt/maverick/detect/'
    defaults['multifits_dir'] = '/work/03946/hetdex/maverick/red1/reductions/'
    defaults['tp_dir'] = '/work/00115/gebhardt/maverick/detect/tp/'
    defaults['norm_dir'] = '/work/00115/gebhardt/maverick/getampnorm/all/'

    defaults['radec_file'] = '/work/00115/gebhardt/maverick/getfib/radec.all'

    defaults['extraction_wl'] = 4505.
    defaults['extraction_wlrange'] = 1035.

    defaults['ifu_search_radius'] = 4.
    defaults['shot_search_radius'] = 600.

    defaults['fitradec_step'] = 0
    defaults['fitradec_nsteps'] = 1
    defaults['fitradec_w_center'] = 4505.
    defaults['fitradec_w_range'] = 3.
    defaults['fitradec_ifit1'] = 1

    defaults['fill'] = 3.
    defaults['sn'] = 8.

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

    parser.add_argument("--extraction_wl", type=float, help="Central "
                        "wavelength for the extraction")
    parser.add_argument("--extraction_wlrange", type=float, help="Wavelength "
                        "range for the extraction")
    parser.add_argument("--radec_file", type=str, help="Filename of file with "
                        "RA DEC PA positions for all shots")

    parser.add_argument("--ifu_search_radius", type=float, help="Radius for "
                        "search for fibers near a given star.")
    parser.add_argument("--shot_search_radius", type=float, help="Radius for "
                        "search for shots near a given star.")

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
        defaults.update(dict(config.items("FitRADEC")))

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h

    # Inherit options from config_paarser
    parser = AP(parents=[conf_parser])

    parser.set_defaults(**defaults)

    parser.add_argument("--logfile", type=str,
                        help="Filename for log file.")

    parser = vext.get_arguments(parser)
    parser = get_arguments(parser)

    # Boolean paramters
    parser.add_argument("--debug", action='store_true',
                        help="Keep temporary directories")

    # positional arguments
    parser.add_argument('ra', metavar='ra', type=float,
                        help='Right Ascension.')
    parser.add_argument('dec', metavar='dec', type=float,
                        help='Declination.')
    parser.add_argument('wl', metavar='wl', type=float,
                        help='Extraction wavelength.')
    parser.add_argument('nightshot', metavar='nightshot', type=str,
                        help='ShotID of observation (e.g. 20180611v021).')
    parser.add_argument('specid', metavar='specid', type=int,
                        help='ID for the spectrum.')

    args = parser.parse_args(remaining_argv)

    args.config_source = config_source
    # should in principle be able to do this with accumulate???
    # args.use_tmp = args.use_tmp == "True"
    # args.remove_tmp = args.remove_tmp == "True"

    # args.fplane_txt = utils.mangle_config_pathname(args.fplane_txt)
    # args.shuffle_cfg = utils.mangle_config_pathname(args.shuffle_cfg)

    args.night, args.shotid = args.nightshot.split('v')

    return args


def fit_radec(args):
    """
    Equivalent of the rsp3f script

    Paramters
    ---------
    args : struct
        The arguments structure
    """

    dithall_file = args.dithall_dir+'/'+args.night + 'v' \
        + args.shotid+'/dithall.use'

    _logger.info('Reading dithall file %s' % dithall_file)
    try:
        dithall = DithAllFile(dithall_file)

    except Exception as e:
        _logger.warn('Failed to read %s' % dithall_file)
        _logger.exception(e)
        return

    _logger.info('Working at #%f %f %f' % (args.ra, args.dec, args.wl))

    wdir = args.wdir

    try:
        starobs, _ = vext.get_star_spectrum_data(args.ra, args.dec, args,
                                                 (args.night, args.shotid),
                                                 False, dithall)

        if not len(starobs):
            raise Exception('No shots found, skipping!')

        # Call rspstar
        # Get fwhm and relative normalizations
        vp.call_getnormexp(args.nightshot, wdir)

        specfiles = \
            vext.extract_star_spectrum(starobs, args,
                                       args.extraction_wl,
                                       args.extraction_wlrange,
                                       wdir)

        vext.get_structaz(starobs, args.multifits_dir)

        vp.run_fitradecsp(args.ra, args.dec, args.fitradec_step,
                          args.fitradec_nsteps, args.fitradec_w_center,
                          args.fitradec_w_range, args.fitradec_ifit1,
                          starobs, specfiles, wdir)

        # Now produce the final output

        if not os.path.exists(os.path.join(wdir, 'spec.out')):
            raise Exception('fitradecsp failed!')

        shutil.move(os.path.join(wdir, 'spec.out'),
                    os.path.join(args.results_dir,
                                 '%s_%d.spec' % (args.nightshot, args.specid)))
        shutil.move(os.path.join(wdir, 'outbest'),
                    os.path.join(args.results_dir,
                                 '%s_%d.res' % (args.nightshot, args.specid)))

    except Exception as e:
        _logger.error(e.message)


def main(jobnum, args):
    """
    Main function.
    """
    # global vdrp_info

    # _logger.info("Executing task : {}".format(task))

    # default is to work in results_dir
    # Create a temporary directory
    wdir = os.path.join(os.getcwd(), args.nightshot + '_' + args.specid)
    _logger.info("Tempdir is {}".format(wdir))
    if not os.path.exists(wdir):
        os.mkdir(wdir)
    _logger.info("Copying over prior data (if any)...")
    # dir_util.copy_tree(results_dir, tmp_dir)
    # set working directory to tmp_dir

    _logger.info("Configuration {}.".format(args.config_source))

    args.wdir = wdir

    # Override the spec flag with the positional parameter
    args.extraction_wl = args.wl

    try:
        _logger.info('Starting spec fit')
        fit_radec(args)
        _logger.info('Finished spec fit')
    except Exception as e:
        _logger.exception(e)

    finally:
        # vdrp_info.save(wdir)
        if not args.debug:
            _logger.info('Removing workdir %s' % wdir)
            shutil.rmtree(wdir, ignore_errors=True)
        _logger.info("Done.")


def fitradec_entrypoint():

    # Here we create another external argument parser, this checks if we
    # are supposed to run in multi-threaded mode.

    # First check if we should loop over an input file
    parser = AP(description='Test', formatter_class=ap_RDHF, add_help=False)
    # parser.add_argument('args', nargs=ap_remainder)
    parser.add_argument('-M', '--multi', help='Input filename to loop over.')
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
    fitradec_entrypoint()
