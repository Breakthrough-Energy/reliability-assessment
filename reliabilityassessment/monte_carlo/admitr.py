import numpy as np


def admitr(BB, NR, NN, BN, LT):
    """
    Remove Ref bus (node)

    :param np.ndarray BB: 2D array of the 'B matrix' (i.e., admittance matrix)
                                  size: # Area-by-# Area
                              BB AT INPUT CONTAINS ENTIRE YBUS MAT
                              BB AT OUTPUT CONTAINS YBUS MAT WITHOUT REF BUS
    :param np.ndarray BN(I,J):
             J=1   BUS NUM
             J=2   LOAD AT BUS I
             J=3   GEN AT BUS I
             J=4   MAX FLOW ALLOWED AT BUS I
    :param int NR: Ref bus
    :param int NN: total numebr of buese (areas)

    :return: (*tuple*) -- a pair of numpy.ndarray objects, i.e.,
             BB: updated B matrix (admittance matrix for DC power flow)
             LT: array recording the orginal bus numebr before the operaiton here

    """

    BT = np.zeros((20, 20))
    LT = np.zeros((20, 1))

    NX = NN - 1
    ii = 0

    for i in range(NN):
        if i == NR:
            continue
        LT[ii] = BN[i, 0]
        ii += 1

        iii = 0
        for j in range(NN):
            if j == NR:
                continue
            BT[ii, iii] = BB[i, j]
            iii += 1

    for i in range(NN):
        for j in range(NN):
            BB[i, j] = 0.0

    for i in range(NX):
        for j in range(NX):
            BB[i, j] = BT[i, j]

    LT[NX + 1] = NR

    return BB, LT
