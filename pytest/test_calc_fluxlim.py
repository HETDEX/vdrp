""" Test the calc_fluxlim module """

from __future__ import absolute_import

from vdrp.calc_fluxlim import compute_apcor

import pytest
import numpy as np
import numpy.random as nr


@pytest.mark.parametrize("add_noise", [True, False])
@pytest.mark.parametrize("shape, nvals", [(10000, 100),
                                           ((10, 10, 10), 20)
                                          ])
def test_compute_apcor(shape, nvals, add_noise):
    """ 
    Test computation of the aperture
    correction by ensuring the
    sorting and selecting largest
    values technique works
    """

    if type(shape) == type(()):
        size = shape[0]*shape[1]*shape[2]
    else:
        size = shape

    vals = np.zeros(size)

    # Fill the first nvals with test value
    # Add a bit of noise to test handling
    # of identical values   
    if add_noise:
        vals[:nvals] = 42 + nr.normal(size=nvals, scale=1e-10)
    else:
        vals[:nvals] = 42

    # Shuffle them up to give the sorting something
    # to do
    nr.shuffle(vals)

    # Reshape to check weird shapes 
    # are supported
    vals = vals.reshape(shape)

    result = compute_apcor(vals, nvals)

    # Ensure test value recovered
    assert result == pytest.approx(42)

