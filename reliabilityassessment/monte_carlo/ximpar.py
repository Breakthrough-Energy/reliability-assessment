import numpy as np


def ximpar(ZB, ZIJ, NI, NN):
    """
    Modifies ZBUS matrix if impedance ZIJ added between bus NI and bus REF

    :param np.array ZB: the input ZBUS matrix  (dim: areas-by-areas)
    :param int NI: from bus id of the newly added branch
    :param int NJ: to bus id of the newly added branch
    :param int NN: total bus number

    return Z -- the modified ZBUS matrix (dim: areas-by-areas)
    """

    ZCOL = np.zeros(NN)
    for i in range(NN):
        ZCOL[i] = ZB[i, NI]

    ZN1 = ZB[NI, NI] - ZIJ

    Z = np.zeros((NN, NN))
    for i in range(NN):
        for j in range(NN):
            Z[i, j] = ZB[i, j] - [ZCOL[i] * ZCOL[j]] / ZN1

    return Z
