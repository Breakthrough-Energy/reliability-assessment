import numpy as np


def wpeakf(DYLOAD):
    """
    Finds weekly peaks (MW)

    :param numpy.array DYLOAD: 2D array, daily peak load amount (MW)
                               1st dim: areaIdx, 2nd dim: dayIdx

    :return: (*numpy.array *) WPEAK -- 2D array, weekly peak load amount (MW)
               1st dim: areaIdx, 2nd dim: weekIdx
    """
    NOAREA = DYLOAD.shape[0]
    WPEAK = np.zeros((NOAREA, 52))
    #  52 weeks in one year is a fixed value in our program

    for areaIdx in range(NOAREA):
        for weekIdx in range(52):
            XMAX = float("-Inf")
            JW = 7
            if (
                weekIdx == 51
            ):  # since 365 day in one year is a fixed value in our program;
                JW = 8  # thus, the 'last week' has '8 days'
            for j in range(JW):
                j1 = weekIdx * 7 + j
                if DYLOAD[areaIdx, j1] > XMAX:
                    XMAX = DYLOAD[areaIdx, j1]
            WPEAK[areaIdx, weekIdx] = XMAX
    return WPEAK
