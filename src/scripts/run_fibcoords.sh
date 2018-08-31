#!/bin/bash

. ../setup.sh

NIGHT=$1
SHOT=$2
CFG=../${NIGHT}v${SHOT}.config
LOG=${NIGHT}v${SHOT}.log
ADD_ARGS=${@:3}

echo $ADD_ARGS

# Fall back to vdrp.config if no night/shot specific 
# configuration file exists.
if [ ! -f $CFG ]; then
    CFG=../vdrp.config
fi

echo Configuration file $CFG
../vdrp/src/python/vdrp/fibcoords.py --logfile $LOG -c $CFG $ADD_ARGS $NIGHT $SHOT