#!/bin/bash

. ../setup.sh

NIGHT=$1
TARGET=$2
CFG=../vdrp.config

../vdrp/src/python/vdrp/fibcoords.py -c $CFG $NIGHT $TARGET
