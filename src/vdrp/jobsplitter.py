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
#SBATCH -p normal              # Queue name
#SBATCH -o {jobname:s}.o%j     # Name of stdout output file
#SBATCH -t {runtime:s}         # Run time (hh:mm:ss)
#SBATCH -A Hobby-Eberly-Telesco
#------------------------------------------------------

#------------------General Options---------------------

cd {workdir:s}
echo " WORKING DIR: {workdir:s}/"

{pyenv:s}
module load gcc
module unload xalt
'''

pyenv = '''export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv shell {pyenv_env:}

'''

pyslurm = '''module load pylauncher
vdrp_runner -c {ncores:d} {debug:s} {runfile}
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


def n_needed(njobs, limit):
    needed = 1
    if njobs > limit:
        needed = njobs / limit
        if njobs % limit:
            needed += 1

    return needed


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

    n_cmds = len(commands)

    # Calculate the number of usable cores per node
    effective_cores_per_node = int(args.cores_per_node / args.cores_per_job)

    # Now the maximum number of jobs per node
    max_jobs_per_node = int(args.max_jobs_per_node * effective_cores_per_node)

    # And the maximum number of jobs per slurm file
    max_jobs_per_file = int(args.nodes * max_jobs_per_node)

    # First find the number of files we need
    n_files = n_needed(n_cmds, max_jobs_per_file)
    # And the resulting number of jobs per file
    jobs_per_file = n_needed(n_cmds, n_files)

    # Next check the number of nodes we need
    n_nodes = n_needed(jobs_per_file, effective_cores_per_node)
    # Now see if we have enough nodes, if we just fill them up
    if n_nodes > args.nodes:
        # We need more nodes than we are allowed to use
        # so limit it to the number of allocated nodes
        n_nodes = args.nodes

    # Finally distribute the jobs over the nodes
    jobs_per_node = n_needed(jobs_per_file, n_nodes)

    print('Found %d commands' % n_cmds)
    print('Splitting them onto %d nodes' % n_nodes)
    print('Running %d jobs per node%s'
          % (jobs_per_node, ' using threading' if args.threading else ''))
    print('Resulting in %d jobfiles' % n_files)

    file_c = 1

    while file_c <= n_files:

        if not len(commands):
            raise Exception('Found fewer commands than expected!')

        cmd_file = '%s_%d%s' % (fname, file_c, fext)
        create_job_file(cmd_file, commands, n_nodes, jobs_per_file,
                        jobs_per_node, args)

        file_c += 1


def create_job_file(fname, commands, n_nodes, jobs_per_file, jobs_per_node,
                    args):

    runtime = args.runtime
    ncores = args.cores_per_job
    if args.threading:
        ncores = args.cores_per_node

    curdir = os.getcwd()

    job_c = 1

    fn, _ = os.path.splitext(fname)

    if args.threading:
        param_c = 1
        parname = '%s_%d.params' % (fn, param_c)
        pf = open(parname, 'w')
        # Count the number of jobs scheduled per node
        node_c = 1

    with open(fname, 'w') as fout:
        # Loop over all jobs that should go into this file
        while job_c <= jobs_per_file:

            if not len(commands):
                break
            # Get the next command
            cmd = commands.pop(0)

            if args.threading:
                # We use threading, so we write a parameter file
                cmd_pars = cmd.split()
                # In case of threading we cannot use individual log files
                if '-l' in cmd_pars:
                    cmd_pars.pop(cmd_pars.index('-l')+1)
                    cmd_pars.pop(cmd_pars.index('-l'))
                pf.write('%s\n' % ' '.join(cmd_pars[1:]))
                node_c += 1

                if node_c > jobs_per_node:
                    # Start a new job file
                    taskname = cmd.split()[0]
                    fout.write('%s -l %s_%d.log --mcores %d -M %s\n'
                               % (taskname, fn, param_c, nthreads, parname))
                    pf.close()
                    param_c += 1
                    parname = '%s_%d.params' % (fn, param_c)
                    pf = open(parname, 'w')
                    node_c = 1
            else:
                fout.write('%s\n' % cmd)

            job_c += 1

        if args.threading:
            # Write the command line in case of threading
            pf.close()
            taskname = cmd.split()[0]
            fout.write('%s -l %s_%d.log --mcores %d -M %s\n'
                       % (taskname, fn, param_c, nthreads, parname))

    # Now write the corresponding slurm file

    with open(fn + '.slurm', 'w') as sf:
        pyenvstr = ''
        if args.py_env is not None:
            pyenvstr = pyenv.format(pyenv_env=args.py_env)

        sf.write(slurm_header.format(pyenv=pyenvstr,
                                     jobname=fn,
                                     nnodes=n_nodes,
                                     ntasks=args.cores_per_node * n_nodes,
                                     runtime=runtime,
                                     workdir=curdir))
        debug = ''
        if args.debug_job:
            debug = '-d'
        sf.write(pyslurm.format(workdir=curdir,
                                ncores=ncores,
                                debug=debug,
                                runfile=fname))
#        else:
#            sf.write(shslurm.format(launcherpath=launcherdir,
#                                    runfile=fname,
#                                    workdir='./'))

        sf.write(slurm_footer)


def getDefaults():
    '''
    Get the defaults for the argument parser. Separating this out
    from the get_arguments routine allows us to use different defaults
    when using the jobsplitter from within a differen script.
    '''
    defaults = {}

    defaults['nodes'] = 5
    defaults['max_jobs_per_node'] = 96
    defaults['cores_per_node'] = 24
    defaults['cores_per_job'] = 1
    defaults['runtime'] = '00:30:00'
    defaults['queue'] = 'normal'

    return defaults


def get_arguments(parser):
    '''
    Add command line arguments for the jobsplitter, this function can be
    called from another tool, adding job splitter support.

    Parameters
    ----------
    parser : argparse.ArgumentParser
    '''

    parser.add_argument('--nodes', '-n', type=int,
                        help='Maximum number of nodes to use per job')
    parser.add_argument('--max_jobs_per_node', '-j', type=int,
                        help='Maximum number of jobs to schedule per node')
    parser.add_argument('--threading', '-t', action='store_true',
                        help='Run one python process per node, and use'
                        'threading.')
    parser.add_argument('--cores_per_node', type=int,
                        help='Number of cores in one node')
    parser.add_argument('--cores_per_job', type=int,
                        help='Number of cores used by one job.')
    parser.add_argument('--runtime', '-r', type=str,
                        help='Expected runtime of slurm job')
    parser.add_argument('--queue', '-q', type=str,
                        help='Slurm queue to use.')
    parser.add_argument('--py_env', '-p', type=str,
                        help='Use a specific pyenv environment')
    parser.add_argument('--debug_job', '-d', action="store_true",
                        help='Keep pylauncher workdir after completion')

    return parser


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

    defaults = getDefaults()

    p.set_defaults(**defaults)

    p = get_arguments(p)

    p.add_argument('cmdfile', type=str, help="""Input commands file""")

    return p.parse_args(args=argv)


def run():

    args = parse_args(sys.argv[1:])
    main(args)


if __name__ == "__main__":

    run()
