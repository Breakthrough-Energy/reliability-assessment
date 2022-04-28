import numpy as np

from reliabilityassessment.monte_carlo.initl import _initl


def test_initl():

    EVNTS = np.zeros(32)
    EVNTS[: 32 - 4 : 4] = np.arange(0, 32 - 4, 4) + 4
    MFA = 0
    IPOINT = -1
    NUMINQ = 0
    JSTEP = 1
    QTR = np.array([13 * 168 + 0.5, 13 * 2 * 168 + 0.5, 13 * 3 * 168 + 0.5])

    ATRIB, CLOCK, IPOINT, MFA, NUMINQ = _initl(JSTEP, EVNTS, IPOINT, MFA, NUMINQ, QTR)

    ATRIB_truth = np.array([1.0, 0.0])
    np.testing.assert_array_almost_equal(ATRIB_truth, ATRIB)

    assert CLOCK == 0.0
    assert IPOINT == 0
    assert MFA == 28
    assert NUMINQ == 7

    EVNTS_truth = np.array(
        [
            20.0,
            0.5,
            2.0,
            0.0,
            8.0,
            2184.5,
            2.0,
            0.0,
            12.0,
            4368.5,
            2.0,
            0.0,
            16.0,
            6552.5,
            2.0,
            0.0,
            -1.0,
            8760.0,
            3.0,
            0.0,
            24.0,
            0.5,
            1.0,
            0.0,
            4.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    np.testing.assert_array_almost_equal(EVNTS_truth, EVNTS)
