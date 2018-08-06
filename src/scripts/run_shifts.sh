#!/bin/bash

. setup.sh

NIGHT=$1
TARGET=$2
RA=$3
DEC=$4
TRACK=1
CFG=vdrp.config

# Either run
vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK
# or
# vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t all
# to run all tasks.

# Or run them one by one:
#vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t cp_post_stamps
#vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t mk_post_stamp_matrix
#vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t rename_cofes
#vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t daophot_find
#vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t daophot_phot_and_allstar
#vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t mktot
#vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t rmaster
#vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t flux_norm
#vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t redo_shuffle
#vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t get_ra_dec_orig
#vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t compute_offset
#vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t add_ifu_xy
#vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t mkmosaic
#

# You can also group task together in a commaseparated list (without spaces)
# this is slightly faster that executing astrometry.py multiple times.
# vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t cp_post_stamps,mk_post_stamp_matrix,daophot_find
