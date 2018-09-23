#!/bin/bash

. ../setup.sh

NIGHT=$1
SHOT=$2
RA=$3
DEC=$4
TRACK=$5
CFG=../${NIGHT}v${SHOT}.config
LOG=${NIGHT}v${SHOT}.log

# Fall back to vdrp.config if no night/shot specific 
# configuration file exists.
if [ ! -f $CFG ]; then
    CFG=../vdrp.config
fi

echo Configuration file $CFG
# Either run
../vdrp/src/python/vdrp/astrometry.py --logfile $LOG -c $CFG $NIGHT $SHOT $RA $DEC $TRACK
# or
# ../vdrp/src/python/vdrp/astrometry.py --logfile $LOG -c $CFG $NIGHT $SHOT $RA $DEC $TRACK -t all
# to run all tasks.

# Or run them one by one:
# ../vdrp/src/python/vdrp/astrometry.py --logfile $LOG -c $CFG $NIGHT $SHOT $RA $DEC $TRACK -t cp_post_stamps
# ../vdrp/src/python/vdrp/astrometry.py --logfile $LOG -c $CFG $NIGHT $SHOT $RA $DEC $TRACK -t mk_post_stamp_matrix
# ../vdrp/src/python/vdrp/astrometry.py --logfile $LOG -c $CFG $NIGHT $SHOT $RA $DEC $TRACK -t rename_cofes
# ../vdrp/src/python/vdrp/astrometry.py --logfile $LOG -c $CFG $NIGHT $SHOT $RA $DEC $TRACK -t daophot_find
# ../vdrp/src/python/vdrp/astrometry.py --logfile $LOG -c $CFG $NIGHT $SHOT $RA $DEC $TRACK -t daophot_phot_and_allstar
# ../vdrp/src/python/vdrp/astrometry.py --logfile $LOG -c $CFG $NIGHT $SHOT $RA $DEC $TRACK -t mktot
# ../vdrp/src/python/vdrp/astrometry.py --logfile $LOG -c $CFG $NIGHT $SHOT $RA $DEC $TRACK -t rmaster
# ../vdrp/src/python/vdrp/astrometry.py --logfile $LOG -c $CFG $NIGHT $SHOT $RA $DEC $TRACK -t flux_norm
#../vdrp/src/python/vdrp/astrometry.py --logfile $LOG -c $CFG $NIGHT $SHOT $RA $DEC $TRACK -t get_ra_dec_orig
#../vdrp/src/python/vdrp/astrometry.py --logfile $LOG  -c $CFG $NIGHT $SHOT $RA $DEC $TRACK -t redo_shuffle
#../vdrp/src/python/vdrp/astrometry.py --logfile $LOG -c $CFG $NIGHT $SHOT $RA $DEC $TRACK -t compute_offset
#../vdrp/src/python/vdrp/astrometry.py --logfile $LOG -c $CFG $NIGHT $SHOT $RA $DEC $TRACK -t compute_with_optimal_ang_off
#../vdrp/src/python/vdrp/astrometry.py --logfile $LOG -c $CFG $NIGHT $SHOT $RA $DEC $TRACK -t combine_radec
#../vdrp/src/python/vdrp/astrometry.py --logfile $LOG -c $CFG $NIGHT $SHOT $RA $DEC $TRACK -t add_ifu_xy
#../vdrp/src/python/vdrp/astrometry.py --logfile $LOG -c $CFG $NIGHT $SHOT $RA $DEC $TRACK -t mkmosaic
#../vdrp/src/python/vdrp/astrometry.py --logfile $LOG -c $CFG $NIGHT $SHOT $RA $DEC $TRACK -t mk_match_plots

# You can also group task together in a commaseparated list (without spaces)
# this is slightly faster that executing astrometry.py multiple times.
# ../vdrp/src/python/vdrp/astrometry.py --logfile $LOG -c $CFG $NIGHT $SHOT $RA $DEC $TRACK -t cp_post_stamps,mk_post_stamp_matrix,daophot_find
