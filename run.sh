#!/bin/bash
# allow python to find vdrp modules
export PYTHONPATH=$PYTHONPATH:/work/04287/mxhf/maverick/sci/panacea/shifts/vdrp/src/python/

# this is required for daophot
module load gcc/5.4.0
# to find daophot
export PATH=/home/00115/gebhardt/lib/daoprogs/:$PATH
# to find daomaster
export PATH=/home/00115/gebhardt/lib/daoprogs/moreprogs2/:$PATH
# to find biwt
export PATH=/home/00115/gebhardt/bin/:$PATH

python vdrp/src/python/vdrp/astrometry.py -c vdrp/config/vdrp.config 20180611 017

