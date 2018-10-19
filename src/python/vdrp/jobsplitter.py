#!/usr/bin/env python

from __future__ import print_function

from argparse import ArgumentParser as AP
from argparse import ArgumentDefaultsHelpFormatter as AHF

import os
import sys


slurm_header = '''#!/bin/bash
#
#------------------Scheduler Options--------------------
#SBATCH -J {jobname:s}         # Job name
#SBATCH -N {nnodes:d}          # Number of nodes
#SBATCH -n {ntasks:d}          # Total number of tasks
#SBATCH -p vis                 # Queue name
#SBATCH -o {jobname:s}.o%j     # Name of stdout output file
#SBATCH -t {runtime:s}         # Run time (hh:mm:ss)
#SBATCH -A Hobby-Eberly-Telesco
#------------------------------------------------------

#------------------General Options---------------------

cd {workdir:s}
echo " WORKING DIR: {workdir:s}/"

module load gcc
module unload xalt
'''

pyslurm = '''module load pylauncher
{launcherpath:s}/vdrprunner.py -c {ncores:d} {debug:s} {runfile}
'''

shslurm = '''module load launcher
export EXECUTABLE=$TACC_LAUNCHER_DIR/init_launcher
export WORKDIR={workdir:s}
export CONTROL_FILE={runfile:s}
export TACC_LAUNCHER_NPHI=0
export TACC_LAUNCHER_PHI_PPN=8
export PHI_WORKDIR=.
export PHI_CONTROL_FILE=phiparamlist
export TACC_LAUNCHER_SCHED=dynamic

$TACC_LAUNCHER_DIR/paramrun SLURM $EXECUTABLE $WORKDIR $CONTROL_FILE $PHI_WORKDIR $PHI_CONTROL_FILE'''

slurm_footer = '''
echo " "
echo " Parameteric VDRP Job Complete"
echo " "
'''


def main(args):

    commands = []

    with open(args.cmdfile, 'r') as f:
        line = f.readline()
        while line != '':
            cmd = line.strip()
            # Skip empty lines and comments
            if cmd != '' and not cmd.startswith('#'):
                commands.append(cmd)
            line = f.readline()

    fname, fext = os.path.splitext(args.cmdfile)

    nc = len(commands)

    nfiles = nc / args.jobs / args.nodes + 1
    jobsperfile = nc / nfiles + 1
    jobspernode = nc / nfiles / args.nodes + 1

    print('Found %d commands' % nc)
    print('Splitting them onto %d nodes' % args.nodes)
    print('Running %d jobs per python instance in %d threads'
          % (jobspernode, args.threads))
    print('Resulting in %d jobfiles' % nfiles)

    file_c = 1

    while file_c <= nfiles:

        if not len(commands):
            raise Exception('Found fewer commands than expected!')

        cmd_file = '%s_%d%s' % (fname, file_c, fext)
        create_job_file(cmd_file, commands, jobsperfile, jobspernode, args)

        file_c += 1


def create_job_file(fname, commands, maxjobs, jobspernode, args):

    runtime = args.runtime
    ncores = args.threads * args.cores
    if ncores > 20:
        print('Would require %d cores, oversubscribing node!')
        ncores = 20
    job_c = 0
    batch_c = 1

    fn, _ = os.path.splitext(fname)

    with open(fname, 'w') as fout:

        subname = '%s.params' % (fn)
        min_t = 0

        with open(subname, 'w') as jf:

            while job_c < maxjobs:
                if not len(commands):
                    break
                cmd = commands.pop(0)
                jf.write('%s\n' % cmd.split(' ', 1)[1])

                if (job_c+1) % jobspernode == 0 or job_c+1 == maxjobs \
                   or len(commands) == 0:
                    taskname = cmd.split()[0]
                    fout.write('%s --mcores %d -M -l %s %s[%d:%d]\n'
                               % (taskname, args.threads,
                                  '%s_%d.log' % (fn, batch_c), subname,
                                  min_t, job_c))
                    batch_c += 1
                    min_t = job_c+1
                job_c += 1

    # Now write the corresponding slurm file

    launcherdir = os.path.dirname(os.path.abspath(__file__))
    with open(fn + '.slurm', 'w') as sf:
        sf.write(slurm_header.format(jobname=fn,
                                     nnodes=args.nodes,
                                     ntasks=20*args.nodes,
                                     runtime=runtime,
                                     workdir='./'))
        debug = ''
        if args.debug:
            debug = '-d'
        sf.write(pyslurm.format(workdir='./',
                                launcherpath=launcherdir,
                                ncores=args.threads*args.cores,
                                debug=debug,
                                runfile=fname))
#        else:
#            sf.write(shslurm.format(launcherpath=launcherdir,
#                                    runfile=fname,
#                                    workdir='./'))

        sf.write(slurm_footer)


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

    p.add_argument('--nodes', '-n', type=int, default=1,
                   help='Number of nodes to use per job')
    p.add_argument('--jobs', '-j', type=int, default=20,
                   help='Number of jobs to schedule per node')
    p.add_argument('--threads', '-t', type=int, default=5,
                   help='Number of threads to use per python process')
    p.add_argument('--cores', '-c', type=int, default=4,
                   help='Number of jobs to schedule per node')
    p.add_argument('--runtime', '-r', type=str, default='00:30:00',
                   help='Expected runtime of slurm job')
    p.add_argument('--queue', '-q', type=str, default='vis',
                   help='Slurm queue to use.')
    p.add_argument('--debug', '-d', action="store_true",
                   help='Keep pylauncher workdir after completion')
    p.add_argument('cmdfile', type=str, help="""Input commands file""")

    return p.parse_args(args=argv)


if __name__ == "__main__":

    args = parse_args(sys.argv[1:])
    main(args)
