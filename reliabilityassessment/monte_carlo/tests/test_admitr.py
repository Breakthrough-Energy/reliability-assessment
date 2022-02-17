import numpy as np

from reliabilityassessment.monte_carlo.admitr import _admitr, admitr


def test_readInputB():
    BB = np.array([[1.5, -0.5, -0.5], [-0.5, 2.5, -1.0], [-0.5, -1.0, 3.0]])
    BN = np.array(
        [[0, 50.0, 100.0, 100.0], [1, 50.0, 100.0, 100.0], [2, 50.0, 100.0, 100.0]]
    )
    NR = 1

    # vanilla version
    BB, LT = _admitr(BB, NR, BN)
    np.testing.assert_array_equal(
        BB, np.array([[1.5, -0.5, 0.0], [-0.5, 3.0, 0.0], [0.0, 0.0, 0.0]])
    )
    np.testing.assert_array_equal(LT, np.array([0.0, 2.0, 1.0]))

    # vectorized version
    BB = np.array([[1.5, -0.5, -0.5], [-0.5, 2.5, -1.0], [-0.5, -1.0, 3.0]])

    BB, LT = admitr(BB, NR, BN)
    np.testing.assert_array_equal(BB, np.array([[1.5, -0.5], [-0.5, 3.0]]))
    np.testing.assert_array_equal(LT, np.array([0.0, 2.0, 1.0]))
