from __future__ import absolute_import

from ..vdrp.photometry import Spectrum, average_spectrum

import pytest

import os


@pytest.fixture(scope="session")
def datadir():
    """ Return a py.path.local object for the test data directory"""
    return os.path.dirname(__file__) + '/data/'


def test_avg_spec(datadir):

    sp = Spectrum()
    sp.read(datadir + 'tmp101.dat')

    a, b = average_spectrum(sp, 4500-100, 4500+100)
    assert round(a, 2) == 425.13
    assert round(b, 4) == 10.8784
