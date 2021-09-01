import numpy as np


def plhr(HRLOAD):
    """
    Find the pool (i.e., "all areas") peak hours.

    :param array HRLOAD: 2D array of hourly load data
                         (0-dim: area index, 1-dim: hour index (1~8760 hr))
    :return: (*np.array*) MXPLHR -- array of the peak hour for the pool

    Note: the hardcoded number "365" and "24" are OK in this code package.
    Because for Monte Carlo Simulation in reliability analysis area, researchers do not
    try to simulate "real-life year", but purely for statistic purpose. In fact,
    each reliability index is eventually an averaged number. So, the leap year is not
    a concern in this area.

    In other words, the reliability indices is an overall measure of the studied
    system. It is not defined for any specific year. For example, researchers will NOT say
    "a reliability index for the year 202O equals XX, for the year 2021 it equals YY, ...".
    """

    # sum across all areas
    summed_load = HRLOAD.sum(axis=0)
    # reshape 8760-length 1D array to N x 24 (N computed automatically)
    reshaped_load = summed_load.reshape((-1, 24))
    # select the index of the greatest value for each day
    daily_peak_hour = reshaped_load.argmax(axis=1)
    # translate hour-of-day to hour-of-year
    MXPLHR = daily_peak_hour + 24 * np.arange(len(daily_peak_hour))

    return MXPLHR
