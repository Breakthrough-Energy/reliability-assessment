import numpy as np

from reliabilityassessment.monte_carlo.ximpar import ximpar


def test_ximpar():

    # NN = 5, the size of Z, ZB matrices (no need now in Python)
    NI = 1 - 1  # be careful: 0-based index in Python
    ZIJ = 0.016667
    ZB = np.array(
        [
            [-0.006667, -0.005000, -0.003333, -0.001667, 0.000000],
            [-0.005000, -0.010000, -0.006667, -0.003333, 0.000000],
            [-0.003333, -0.006667, -0.010000, -0.005000, 0.000000],
            [-0.001667, -0.003333, -0.005000, -0.006667, 0.000000],
            [0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
        ]
    )

    Z = ximpar(ZB, ZIJ, NI)

    Z_true = np.array(
        [
            [-0.004762, -0.003571, -0.002381, -0.001190, 0.000000],
            [-0.003571, -0.008929, -0.005952, -0.002976, 0.000000],
            [-0.002381, -0.005952, -0.009524, -0.004762, 0.000000],
            [-0.001190, -0.002976, -0.004762, -0.006548, 0.000000],
            [0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
        ]
    )

    np.testing.assert_array_almost_equal(Z, Z_true)
