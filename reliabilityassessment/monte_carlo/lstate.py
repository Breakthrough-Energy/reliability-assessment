import numpy as np


def lstate(PROBL, test_seed=None):
    """Establishes the line state according the probability of the line being in
    each particular state. This implementation assumes the lines can be in up to 6 states overall.

    :param int NLINES: total number of lines
    :param numpy.ndarray PROBL: array of the cumulative capacity probability of each line
    :raises TypeError:
        NLINES must be an integer
        PROBL must be a numpy array
    :raises ValueError: NLINES must be positive
    :raises Exception: PROBL values must sum to 1
    :return: (*numpy.ndarray*) -- array of line state value (in integer) of each line
    """

    if not isinstance(PROBL, np.ndarray):
        raise TypeError("PROBL must be a numpy array")

    NLINES = PROBL.shape[1]

    # set random seed for reproducibility
    if test_seed is not None:
        rng = np.random.default_rng(seed=test_seed)
    else:
        rng = np.random.default_rng()

    LNSTAT = np.zeros(NLINES, dtype=np.int32)
    for i in range(NLINES):
        # Draw a random number for each line
        RANDL = rng.random()
        # RANDL = np.random.random()

        # Determine the State and Available Capacity for Each Generator.
        if RANDL <= PROBL[0, i]:
            LNSTAT[i] = 1
        elif RANDL <= PROBL[1, i]:
            LNSTAT[i] = 2
        elif RANDL <= PROBL[2, i]:
            LNSTAT[i] = 3
        elif RANDL <= PROBL[3, i]:
            LNSTAT[i] = 4
        elif RANDL <= PROBL[4, i]:
            LNSTAT[i] = 5
        else:
            LNSTAT[i] = 6
    return LNSTAT
