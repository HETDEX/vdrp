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
import astropy.stats as aps
# from astropy.io import ascii
import numpy as np

import vdrp.mplog as mplog
import vdrp.utils as utils
from vdrp.containers import DithAllFile
import vdrp.jobsplitter as vj


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

    defaults['logfile'] = 'fluxlim.log'

    defaults['dithall_dir'] = '/work/00115/gebhardt/maverick/detect/'

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

    # Now setup the defaults for the jobsplitter
    job_defaults = vj.getDefaults()

    # Update them for fluxlim
    job_defaults['cores'] = 1
    job_defaults['threads'] = 24

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h

    # Inherit options from config_paarser
    parser = AP(parents=[conf_parser])

    parser.set_defaults(**defaults)
    parser.set_defaults(**job_defaults)

    parser.add_argument("--logfile", type=str,
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

    # Add the jobsplitter arguments
    vj.get_arguments(parser)

    args, remaining_args = parser.parse_known_args(remaining_argv)
    # args = parser.parse_args(remaining_argv)

    args.config_source = config_source
    # should in principle be able to do this with accumulate???
    # args.use_tmp = args.use_tmp == "True"
    # args.remove_tmp = args.remove_tmp == "True"

    # NEW set the bin_dir to the vdrp supplied bin directory
    # args.bin_dir = utils.bindir()

    # args.fplane_txt = utils.mangle_config_pathname(args.fplane_txt)
    # args.shuffle_cfg = utils.mangle_config_pathname(args.shuffle_cfg)

    return args, remaining_args


def setup_fluxlim(args, rargs):
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

    fname = 'flim%s' % nightshot

    with open(fname, 'w') as f:

        for ifu in ifus:
            ifu_dith = dithall.where(dithall.ifuslot == ifu)
            dist = np.sqrt(ifu_dith.x*ifu_dith.x + ifu_dith.y*ifu_dith.y)
            sortidx = np.argsort(dist)

            ra_mean = aps.biweight_location(ifu_dith.ra[sortidx][0:2])
            dec_mean = aps.biweight_location(ifu_dith.dec[sortidx][0:2])

            ixyname = ifu_dith.filename[sortidx][0]

            f.write('vdrp_calc_flim %s %.7f %.7f %s %s %s\n'
                    % (' '.join(rargs), ra_mean, dec_mean, args.night,
                       args.shotid, '_'.join(ixyname.split('_')[0:4])))

    # Now prepare the job splitter for it
    args.cmdfile = fname

    vj.main(args)


def setup_fluxlim_entrypoint():
    '''
    Entrypoint to run the flux limit calculation for one night / shot
    combination

    '''
    # Here we create another external argument parser, this checks if we
    # are supposed to run in multi-threaded mode.

    args, rargs = parseArgs(sys.argv[1:])

    mplog.setup_mp_logging(args.logfile)

    # Create results directory for given night and shot
    cwd = _baseDir
    results_dir = os.path.join(cwd, args.night + 'v' + args.shotid,  'flim')
    utils.createDir(results_dir)
    args.results_dir = results_dir

    # save arguments for the execution
    # default is to work in results_dir
    wdir = results_dir

    _logger.info("Configuration {}.".format(args.config_source))

    args.curdir = os.path.abspath(os.path.curdir)
    args.wdir = wdir

    try:
        os.chdir(wdir)

        _logger.info('Starting flux limit setup')
        setup_fluxlim(args, rargs)
        _logger.info('Finished flux limit setup')
    except Exception as e:
        _logger.exception(e)

    finally:
        os.chdir(args.curdir)
        _logger.info("Done.")


if __name__ == "__main__":
    setup_fluxlim_entrypoint()
