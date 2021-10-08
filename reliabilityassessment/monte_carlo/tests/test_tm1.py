import numpy as np

from reliabilityassessment.monte_carlo.tm1 import tm1


def test_tm1():
    # NOAREA = 3, NLINES = 3
    LP = np.array([[0, 0, 1], [1, 1, 2], [2, 2, 0]], dtype=int)
    BLP = np.array(
        [[-120.0, 300.0, 300.0], [-60.0, 150.0, 150.0], [-80.0, 100.0, 100.0]]
    )
    BN = np.array(
        [
            [0, 50.0, 100.0, 100.0],
            [1, 50.0, 100.0, 100.0],
            [2, 50.0, 100.0, 100.0],
        ]  # (NOAREA,4)
    )

    NR = 0
    BLP0_ = np.array([-120.0, -60.0, -80.0])
    BB_ = np.array([[-180.0, 60.0], [60.0, -140.0]])
    LT_ = np.array([1, 2, 0], dtype=int)
    ZB_ = np.linalg.inv(BB_)
    BLP0, BB, LT, ZB = tm1(BN, LP, BLP, NR)
    np.testing.assert_array_equal(BLP0_, BLP0)
    np.testing.assert_array_equal(BB_, BB)
    np.testing.assert_array_equal(LT_, LT)
    np.testing.assert_array_equal(ZB_, ZB)

    NR = 1
    BLP0_ = np.array([-120.0, -60.0, -80.0])
    BB_ = np.array([[-200.0, 80.0], [80.0, -140.0]])
    LT_ = np.array([0, 2, 1], dtype=int)
    ZB_ = np.linalg.inv(BB_)
    BLP0, BB, LT, ZB = tm1(BN, LP, BLP, NR)
    np.testing.assert_array_equal(BLP0_, BLP0)
    np.testing.assert_array_equal(BB_, BB)
    np.testing.assert_array_equal(LT_, LT)
    np.testing.assert_array_equal(ZB_, ZB)
