import numpy as np


def ximpa(ZB, ZIJ, NI, NJ, NN):
    """
    Modifies ZBUS matrix if ZIJ added between bus NI,NJ

    :param np.array ZB: the input ZBUS matrix  (dim: areas-by-areas)
    :param int ZIJ: the impedance to be added to the system between bus NI,NJ
    :param int NI: from bus id of the newly added branch (0-based index)
    :param int NJ: to bus id of the newly added branch (0-based index)
    :param int NN: total bus number
    :return: (*numpy.ndarray*) -- the modified ZBUS matrix (dim: areas-by-areas)
    """

    ZCOL = np.zeros(NN)
    for i in range(NN):
        ZCOL[i] = ZB[i, NI] - ZB[i, NJ]

    ZN1 = ZB[NI, NI] + ZB[NJ, NJ] - 2.0 * ZB[NI, NJ] - ZIJ

    Z = np.zeros((NN, NN))
    for i in range(NN):
        for j in range(NN):
            Z[i, j] = ZB[i, j] - (ZCOL[i] * ZCOL[j]) / ZN1

    return Z
