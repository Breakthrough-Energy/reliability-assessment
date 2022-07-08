import numpy as np

from reliabilityassessment.data_processing.smaint import smaint


def pmsc_mock(areaIdx, ID):
    ID[:] = np.array([[1, 1, 1, 0, 3, 0, 0, 0], [0, 0, 0, 1, 4, 0, 0, 0]], dtype=int)
    return


def test_smaint(mocker):
    mocker.patch(
        "reliabilityassessment.data_processing.smaint.pmsc", side_effect=pmsc_mock
    )

    NOAREA = 3

    ID = np.zeros((2, 8), dtype=int)
    NUNITS = ID.shape[0]

    JPLOUT = smaint(NOAREA, ID)

    JPLOUT_true = (-1) * np.ones((52, 1 + NUNITS), dtype=int)
    JPLOUT_true[:, 0] = 0
    JPLOUT_true[:5, 0] = [1, 2, 2, 1, 1]
    JPLOUT_true[:5, 1] = [1, 1, 1, 0, 0]
    JPLOUT_true[1:3, 2] = 0

    np.testing.assert_almost_equal(JPLOUT_true, JPLOUT)
