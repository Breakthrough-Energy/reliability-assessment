import numpy as np

from reliabilityassessment.monte_carlo.flo import flo


def test_flo():
    # NOAREA = 3
    # NLINES = 3
    # SFLOW = np.zeros((NOAREA,))
    # FLOW = np.zeros((NLINES,))
    LP = np.array([[0, 0, 1], [1, 1, 2], [2, 2, 0]], dtype=int)
    BLP = np.array([[-120, 300, 300], [-60, 150, 150], [-80, 100, 100]])
    THET = np.array([0.1, 0.2, 0.3])  # shape: (NOAREA,)

    FLOW_ = np.array([12.0, 6.0, -16.0])
    SFLOW_ = np.array([-28.0, 6.0, 22.0])

    SFLOW, FLOW = flo(LP, BLP, THET)
    np.testing.assert_array_almost_equal(FLOW, FLOW_)
    np.testing.assert_array_almost_equal(SFLOW, SFLOW_)

    # NOAREA = 3
    # NLINES = 2
    # SFLOW = np.zeros((NOAREA,))
    # FLOW = np.zeros((NLINES,))
    LP = np.array([[0, 0, 1], [1, 2, 0]], dtype=int)
    BLP = np.array([[-120, 300, 300], [-80, 100, 100]])
    THET = np.array([0.1, 0.2, 0.3])  # shape: (NOAREA,)

    FLOW_ = np.array([12.0, -16.0])
    SFLOW_ = np.array([-28.0, 12.0, 16.0])

    SFLOW, FLOW = flo(LP, BLP, THET)
    np.testing.assert_array_almost_equal(FLOW, FLOW_)
    np.testing.assert_array_almost_equal(SFLOW, SFLOW_)
