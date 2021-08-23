import numpy as np


def gstate(PROBG, RATING, DERATE, PLNDST):
    """
    It samples the state of available capacity for each generator up to NUNITS

    :param numpy.ndarray PROBG: 2D array of accumulated probability for each capacity tier of each unit
    :param numpy.ndarray DERATE: array of the derated capacity of each unit
    :param numpy.ndarray RATING:  array of the fully rated capacity of each unit
    :param numpy.ndarray PLNDST: array of the on/off status (due to planned maintenance) of each unit

    :return: :return: (*tuple*) -- a pair of numpy.ndarray objects for unadjusted power capacity in p.u.
                                   of each unit and finalized power capacity in MW of each unit.
    """

    NUNITS = len(PROBG)
    print("Enter function gstate, total number of Gen units is %d" % (NUNITS))

    PCTAVL = np.zeros((NUNITS, 1))
    AVAIL = np.zeros((NUNITS, 1))

    for i in range(NUNITS):
        # Draw a random number for each unit
        rand_number = np.random.rand()

        # Determine the State and Available Capacity for Each Generator.
        if rand_number <= PROBG[i, 0]:
            PCTAVL[i] = 1.0  # normalized value for gen. capacity is used here
        elif rand_number <= PROBG[i, 1]:
            PCTAVL[i] = DERATE[i]  # normalized value for gen. capacity is used here
        else:
            PCTAVL[i] = 0

        AVAIL[i] = PCTAVL[i] * RATING[i] * PLNDST[i]

    return PCTAVL, AVAIL
