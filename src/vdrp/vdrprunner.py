#!/usr/bin/env python

from __future__ import print_function

from argparse import ArgumentParser as AP
from argparse import ArgumentDefaultsHelpFormatter as AHF

import pylauncher
import platform
import shutil
import os
import sys


def VDRPLauncher(commandfile, **kwargs):
    """A LauncherJob for a file of single or multi-threaded commands.

    The following values are specified for your convenience:

    * hostpool : based on HostListByName
    * commandexecutor : SSHExecutor
    * taskgenerator : based on the ``commandfile`` argument
    * completion : based on a directory ``pylauncher_tmp`` with
      jobid environment variables attached

    :param commandfile: name of file with commandlines (required)
    :param cores: number of cores (keyword, optional, default=1)
    :param workdir: directory for output and temporary files
        (optional, keyword, default uses the job number); the launcher
         refuses to reuse an already existing directory
    :param debug: debug types string (optional, keyword)
    """
    jobid = pylauncher.JobId()
    debug = kwargs.pop("debug", "")
    workdir = kwargs.pop("workdir", "pylauncher_tmp"+str(jobid))
    cores = kwargs.pop("cores", 1)

    hosttag = ".wrangler.tacc.utexas.edu"
    if 'maverick' in platform.node():
        hosttag = ".maverick.tacc.utexas.edu"
    if 'stampede2' in platform.node():
        hosttag = ".stampede2.tacc.utexas.edu"

    job = pylauncher.LauncherJob(
        hostpool=pylauncher.HostPool(
            hostlist=pylauncher.SLURMHostList(
                tag=hosttag),
            commandexecutor=pylauncher.SSHExecutor(
                workdir=workdir, debug=debug),
            debug=debug),
        taskgenerator=pylauncher.TaskGenerator(
            pylauncher.FileCommandlineGenerator(commandfile, cores=cores,
                                                debug=debug),
            completion=lambda x: pylauncher.FileCompletion(taskid=x,
                                                           stamproot="expire",
                                                           stampdir=workdir),
            debug=debug),
        debug=debug, **kwargs)
    print('Running job')
    print(job)
    job.run()
    print(job.final_report())


def main(args):
    # pylauncher.ClassicLauncher(args.cmdfile, debug="job+host+task",
    #                            cores=args.cores)
    workdir = 'pylauncher_vdrp'
    if os.path.exists(workdir):
        shutil.rmtree(workdir)

    debug = ''
    if args.debug:
        print('Enabling debugging')
        debug = 'job+host+task+exec+command'
    VDRPLauncher(args.cmdfile, cores=args.cores,
                 workdir=workdir, debug=debug)

    if not args.debug:
        shutil.rmtree(workdir)


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


def run():

    print('Starting vdrp runner')
    args = parse_args(sys.argv[1:])
    main(args)


if __name__ == "__main__":

    run()
