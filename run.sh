#!/bin/bash

. setup.sh

NIGHT=20180611
TARGET=017
RA=13.8447
DEC=51.3479
TRACK=1
CFG=vdrp.config

# Either run
# python vdrp/src/python/vdrp/astrometry.py -c vdrp.config $NIGHT $TARGET $RA $TRACK 1
# or
# python vdrp/src/python/vdrp/astrometry.py -c vdrp.config $NIGHT $TARGET $RA $TRACK $TRACK -t all
# to run all tasks.

# Or run them one by one:
#python vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t cp_post_stamps
#python vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t mk_post_stamp_matrix
#python vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t rename_cofes
#python vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t daophot_find
#python vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t daophot_phot_and_allstar
#python vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t mktot
#python vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t rmaster
#python vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t flux_norm
#python vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t redo_shuffle
#python vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t get_ra_dec_orig
python vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t compute_offset
#python vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t add_ifu_xy
#python vdrp/src/python/vdrp/astrometry.py -c $CFG $NIGHT $TARGET $RA $DEC $TRACK -t mkmosaic
