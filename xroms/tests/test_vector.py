import numpy as np
import xarray as xr

import xroms


def test_rotate_vectors():

    # angle from x axis
    u, v = 1, 0
    assert (u, v) == xroms.rotate_vectors(u, v, 0, isradians=True, reference="xaxis")
    assert np.allclose(
        (0, 1), xroms.rotate_vectors(u, v, 90, isradians=False, reference="xaxis")
    )

    # comparisons
    assert np.allclose(
        xroms.rotate_vectors(u, v, 180, isradians=False, reference="compass"),
        xroms.rotate_vectors(u, v, 180, isradians=False, reference="xaxis"),
    )
    assert np.allclose(
        xroms.rotate_vectors(u, v, np.pi / 2, isradians=True, reference="xaxis"),
        xroms.rotate_vectors(u, v, 90, isradians=False, reference="xaxis"),
    )

    # angle from compass
    assert (u, v) == xroms.rotate_vectors(u, v, 0, isradians=True, reference="compass")
    assert np.allclose(
        (0, -1), xroms.rotate_vectors(u, v, 90, isradians=False, reference="compass")
    )
