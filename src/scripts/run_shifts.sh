#!/bin/bash

. ../setup.sh

NIGHT=$1
SHOT=$2
RA=$3
DEC=$4
TRACK=$5
CFG=../${NIGHT}v${SHOT}.config
LOG=${NIGHT}v${SHOT}.log

echo $ADD_ARGS

# Fall back to vdrp.config if no night/shot specific 
# configuration file exists.
if [ ! -f $CFG ]; then
    CFG=../vdrp.config
fi

echo Configuration file $CFG
../vdrp/src/python/vdrp/astrometry.py --logfile $LOG -c $CFG $NIGHT $SHOT $RA $DEC $TRACK
