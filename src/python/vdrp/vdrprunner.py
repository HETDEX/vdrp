#!/usr/bin/env python


from __future__ import print_function

from argparse import ArgumentParser as AP
from argparse import ArgumentDefaultsHelpFormatter as AHF

import tempfile
import pylauncher
import os
import sys


def main(args):
    # pylauncher.ClassicLauncher(args.cmdfile, debug="job+host+task",
    #                            cores=args.cores)
    workdir = 'pylauncher_vdrp'
    if os.path.exists(workdir):
        os.rmdir(workdir)
    pylauncher.ClassicLauncher(args.cmdfile, cores=args.cores, workdir=workdir)

    if not args.debug:
        os.rmdir(workdir)


def parse_args(argv):
    """
    Command line parser

    Parameters
    ----------
    argv : list of strings
        list to parsed

    Returns
    -------
    namespace:
         Parsed arguments
    """

    p = AP(formatter_class=AHF)

    p.add_argument('--cores', '-c', type=int, default=1,
                   help='Number of cores for multiprocessing')
    p.add_argument('--debug', '-d', action="store_true",
                   help='Keep pylauncher workdir after completion')
    p.add_argument('cmdfile', type=str, help="""Input commands file""")

    return p.parse_args(args=argv)


if __name__ == "__main__":

    args = parse_args(sys.argv[1:])
    main(args)
