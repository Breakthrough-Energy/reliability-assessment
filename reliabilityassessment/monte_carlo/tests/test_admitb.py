import numpy as np

from reliabilityassessment.monte_carlo.admitb import admitb


def test_admitb():
    NOAREA = 3
    # NLINES = 3
    LP = np.array([[0, 0, 1], [1, 1, 2], [2, 2, 0]], dtype=int)
    BLP = np.array([[-120, 300, 300], [-60, 150, 150], [-80, 100, 100]])
    BB_ = np.array([[-200.0, 120, 80], [120, -180, 60], [80, 60, -140]])
    BB = np.zeros((NOAREA, NOAREA))
    admitb(LP, BLP, BB)
    np.testing.assert_array_almost_equal(BB, BB_)
