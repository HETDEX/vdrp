#!/bin/bash

. setup.sh

python -m unittest tests.test_cltools tests.test_daophot
