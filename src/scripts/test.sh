#!/bin/bash

# . ../setup.sh

# python -m unittest vdrp.tests.test_cltools vdrp.tests.test_daophot  vdrp.tests.test_astrometry

cd unittest
python runtests.py
