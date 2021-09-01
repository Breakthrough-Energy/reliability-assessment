from bisect import bisect_left

import numpy as np


def gstate(PROBG, RATING, DERATE, PLNDST, rng=np.random.default_rng()):
    """
    It samples the state of available capacity for each generator up to NUNITS

    :param numpy.ndarray PROBG: 2D array of accumulated probability for each capacity tier of each unit
    :param numpy.ndarray DERATE: array of the derated capacity of each unit
    :param numpy.ndarray RATING:  array of the fully rated capacity of each unit
    :param numpy.ndarray PLNDST: array of the on/off status (due to planned maintenance) of each unit

    :return: (*tuple*) -- a pair of numpy.ndarray objects for unadjusted power capacity in p.u.
                                   of each unit and finalized power capacity in MW of each unit.
    """

    NUNITS = len(PROBG)
    print("Enter function gstate, total number of Gen units is %d" % (NUNITS))

    PCTAVL = np.zeros((NUNITS, 1))
    AVAIL = np.zeros((NUNITS, 1))

    # Vectorized operation utilizing the bisect API
    aux = {i: {0: 1, 1: d, 2: 0} for i, d in enumerate(DERATE)}
    PCTAVL = np.array(
        [aux[i][bisect_left(PROBG[i], x)] for i, x in enumerate(rng.random(len(PROBG)))]
    )
    AVAIL = (PCTAVL * RATING * PLNDST).reshape(-1, 1)

    return PCTAVL, AVAIL
