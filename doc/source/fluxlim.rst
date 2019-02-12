Fluxlimit routines
==================

Setting up and running the fluxlimit calculations
*************************************************

To calculate the fluxlimit cube of a given night shot call:

``vdrp_setup_flim night shot``

This will create a subdirectory tree of the form ``nightvshot/flim``
and in there a slurm batch script named ``flimnightvshot.slurm`` and
the corresponding input files. Running the script as

``vdrp_setup_flim --commit night shot``

the slurm script will be sent to the batch system automatically.
If needed the default runtime of ``06:00:00`` can be modified
using --runtime on the command line.
