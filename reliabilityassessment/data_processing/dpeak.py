import numpy as np


def _dpeak(HRLOAD, hour_within_day=24):
    """
    Find daily peaks and hour of daily peaks

    NOAREA: total number of areas
    HRLOAD: 2D array of annual (hourly) load profile

    return (*tuple*) -- a tuple of multiple arrays, i.e.,
        DYLOAD: 2D array, daily peak load amount (MW)
        MAXHR: 2D array, daily peak load hour index
        (not in the range of 1 to 24 but in the range 1 to 8760;
        also note that python index starts from 0)

    """
    NOAREA = HRLOAD.shape[0]
    day_within_year = HRLOAD.shape[1] // hour_within_day
    DYLOAD = np.zeros((NOAREA, day_within_year))
    MAXHR = np.zeros((NOAREA, day_within_year), dtype=int)

    for areaIdx in range(NOAREA):
        for dayIdx in range(day_within_year):
            XMAX = float("-inf")
            for j in range(hour_within_day):
                j1 = dayIdx * hour_within_day + j
                if HRLOAD[areaIdx, j1] <= XMAX:
                    continue
                XMAX = HRLOAD[areaIdx, j1]
                MAXHR[areaIdx, dayIdx] = j1
            DYLOAD[areaIdx, dayIdx] = XMAX
    return MAXHR, DYLOAD


def dpeak(HRLOAD, hour_within_day=24):
    """
    Find daily peaks and hour of daily peaks

    NOAREA: total number of areas
    HRLOAD: 2D array of annual (hourly) load profile

    return (*tuple*) -- a tuple of multiple arrays, i.e.,
        DYLOAD: 2D array, daily peak load amount (MW), in the shape of ((NOAREA, 365))
        MAXHR: 2D array, daily peak load hour index, in the shape of ((NOAREA, 365))
        (not in the range of 1 to 24 but in the range 1 to 8760;
        also note that python index starts from 0)

    """
    NOAREA = HRLOAD.shape[0]

    # reshape (area, hour_within_year) to (area, hour_within_day, day_within_year)
    # note: the 'order' parameter should be 'F' to keep the original internal ordering.
    reshaped = HRLOAD.reshape((NOAREA, hour_within_day, -1), order="F")
    day_within_year = reshaped.shape[2]
    #  i.e., HRLOAD .shape[1] / hour_within_day
    DYLOAD = reshaped.max(axis=1)
    MAXHR = reshaped.argmax(axis=1) + hour_within_day * np.arange(day_within_year)

    return MAXHR, DYLOAD
