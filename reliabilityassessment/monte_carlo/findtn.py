import numpy as np


def findtn(JENT, INTCH, JDAY):
    """
    Determine (net) inter-area transferred power on a secific JDAY
    for an interested  pair of areas based on fixed contracts (if there exists)

    :param np.arrays JENT: 2D array of pointer identifying a specific contract
                            , of the shape (NOAREA, NOAREA)
                            ,the (i,j)-th element means the contract (no.) between area-i and area-j

    :param np.arrays INTCH: 2D array of the contract content
                            , of the shape (total numebr of contracts, 365)
                            , the (JENT[i,j], k)-th element means the contracted power (MW) on the kth day
                            from area-i to area-j

    :param int JDAY: integer index for the day

    :return: (*np.arrays*)  TRNSFR: 1D array, of the shape (NOAREA,)
                             the ith element means the power transerred-out from this area-i to the rest of pool
                             **TRANSFERS treated as from higher to lower area number**
    """
    print("Entering function findtn")

    NOAREA = JENT.shape[0]
    TRNSFR = np.zeros((NOAREA,))
    for j2 in range(NOAREA):
        for j1 in range(NOAREA):
            if JENT[j1, j2] > -1:
                JPOINT = JENT[j1, j2]
                TRNSFR[j1] = TRNSFR[j1] + INTCH[JPOINT, JDAY]
                TRNSFR[j2] = TRNSFR[j2] - INTCH[JPOINT, JDAY]
    return TRNSFR
