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

    defaults = getDefaults()

    # Now setup the defaults for the jobsplitter
    job_defaults = vj.getDefaults()

    # Update them for fluxlim
    job_defaults['cores_per_job'] = 2
    job_defaults['nodes'] = 5
    job_defaults['runtime'] = '06:00:00'

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h

    # Inherit options from config_paarser
    parser = AP(description=__doc__,  # printed with -h/--help
                # Don't mess with format of description
                formatter_class=ap_RDHF,
                # Turn off help, so we print all options in response to -h
                add_help=False)

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

    args, remaining_args = parser.parse_known_args(argv)
    # args = parser.parse_args(remaining_argv)

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

            logstr = ''
            if '-l' not in rargs:
                logstr = '-l %s_%s_%s.log'  \
                    % (args.night, args.shotid,
                       '_'.join(ixyname.split('_')[0:4]))

            f.write('vdrp_calc_flim %s %s %.7f %.7f %s %s %s\n'
                    % (' '.join(rargs), logstr, ra_mean, dec_mean, args.night,
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

    mplog.setup_mp_logging(args.logfile, 'INFO')

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
