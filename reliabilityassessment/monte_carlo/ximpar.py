import numpy as np


def ximpar(ZB, ZIJ, NI, NN):
    """
    Modifies ZBUS matrix if impedance ZIJ added between bus NI and REF

    :param np.array ZB: the input ZBUS matrix  (dim: areas-by-areas)
    :param int ZIJ: the impedance to be added to the system
    :param int NI: from bus id of the newly added branch
    :param int NN: total bus number
    :return: (*numpy.ndarray*) -- the modified ZBUS matrix (dim: areas-by-areas)
    """

    ZCOL = np.zeros(NN)
    for i in range(NN):
        ZCOL[i] = ZB[i, NI]

    ZN1 = ZB[NI, NI] - ZIJ

    Z = np.zeros((NN, NN))
    for i in range(NN):
        for j in range(NN):
            Z[i, j] = ZB[i, j] - (ZCOL[i] * ZCOL[j]) / ZN1

    return Z
