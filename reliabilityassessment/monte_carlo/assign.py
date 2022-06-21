import numpy as np


def assign(INJ, LOD, LT, NN, NLS):
    """
    Assigns injections at buses so as to balance positive and negative injections

    :param numpy.ndarray INJ: 1D array with shape
                              vector of input injections

    :param numpy.ndarray LOD: 1D array with shape (NLINES, 3)
                              vector of loads

    :param numpy.ndarray LT: 1D array with shape (NN, ) (i.e., NOAREA )
                             LT[I] contains the actual node index corresponding
                             to node I of the reduced admittance matrix here;

    :param int NN: user-specified dimension for the arrays here

    :param int NLS: indicator of loss sharing mode,
                    0    LOSS SHARING
                    1    NO LOSS SHARING

    :return: (*numpy.ndarray*) -- INJ1: 2D array with shape (NN, ) (i.e., NOAREA )
                                        vector of modified injections
                                  LODC: 2D array with shape (NN, ) (i.e., NOAREA )
                                        vector of load curtailments

    """

    INJ1 = np.zeros(NN)
    LODC = np.zeros(NN)

    for i in range(NN):
        INJ1[i] = INJ[i]

    SUMP = 0.0
    SUMN = 0.0

    for i in range(NN):
        if INJ1[i] <= 0:
            SUMN += INJ1[i]
            continue
        SUMP = SUMP + INJ1[i]

    SUMN = abs(SUMN)
    if SUMN <= SUMP:
        for i in range(NN):
            if INJ1[i] <= 0:
                continue
            INJ1[i] = INJ1[i] * SUMN / SUMP
        return INJ1, LODC

    if NLS != 0:
        for i in range(NN):
            if INJ1[i] > 0:
                continue
            PINT = INJ1[i]
            INJ1[i] = INJ1[i] * SUMP / SUMN
            J = LT[i]
            LODC[J] = INJ1[i] - PINT
        return INJ1, LODC

    SUML = 0.0
    for i in range(NN):
        SUML = SUML + LOD[i]

    DIFF = SUMN - SUMP
    for i in range(NN):
        J = LT[i]
        LODC[J] = LOD[J] * DIFF / SUML
        INJ1[i] = INJ1[i] + LODC[J]

    return INJ1, LODC
