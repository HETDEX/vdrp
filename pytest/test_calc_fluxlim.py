""" Test the calc_fluxlim module """

from __future__ import absolute_import

from ..vdrp.calc_fluxlim import compute_apcor

import pytest
import numpy as np
import numpy.random as nr


@pytest.mark.parameterize("shape, nvals", [(10000, 100),
                                           ((10, 10, 10), 20)
                                          ])
def test_compute_apcor(shape, nvals):
    """ 
    Test computation of the aperture
    correction by ensuring the
    sorting and selecting largest
    values technique works
    """

    size = shape[0]*shape[1]*shape[2]
    vals = np.zeros(size)

    # Fill the first nvals with test value
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

