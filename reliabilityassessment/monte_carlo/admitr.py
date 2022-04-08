import numpy as np


# vanilla version
def _admitr(BB, BN, NR):
    """
    Remove Ref bus (node) from the admittance matrix

    :param numpy.ndarray BB: 2D array with shape (NOAREA, NOAREA), B Matrix (i.e.
                             admittance matrix) in DC power flow
    :param numpy.ndarray BN: 2D array with shape (NOAREA, 4)
                             BN[I, 0]: area (bus/node) number
                             BN[I, 1]: area load, overridden by net injection after
                                       calling :py:func: `tm2`
                             BN[I, 2]: area net injection, overridden by load
                                       curtailment after calling :py:func: `tm2`
                             BN[I, 3]: area constraint on total power flow
    :param int NR: index number of the reference node.
    :return: (*tuple*) -- BB: 2D array with shape (NOAREA, NOAREA)
                              B Matrix without reference bus in DC power flow
                          LT: 1D array with shape (NOAREA, )
                              original node index of reduced B matrix, ``BB``
    """

    NN = BB.shape[0]  # total numebr of buses (i.e., areas)
    NX = NN - 1

    BT = np.zeros((NX, NX))
    LT = np.zeros((NN,))

    ii = 0
    for i in range(NN):
        if i == NR:
            continue
        LT[ii] = BN[i, 0]
        iii = 0
        for j in range(NN):
            if j == NR:
                continue
            BT[ii, iii] = BB[i, j]
            iii += 1
        ii += 1
    LT[NX] = NR

    BB[:NX, :NX] = BT[:, :]
    BB[-1, :] = 0
    BB[0:, -1] = 0

    return BB, LT


# vectorized version
def admitr(BB, BN, NR):
    """
    Remove Ref bus (node) from the admittance matrix

    :param numpy.ndarray BB: 2D array of the 'B matrix' (i.e., admittance matrix)
                                  size: # Area-by-# Area
                          at input stage, BB  contains the entire YBUS matrix
                          at output stage, BB  contains YBUS matrix without the REF bus
    :param numpy.ndarray BN(I,J): for bus (i.e., area) I,
             J=0   bus number
             J=1   load (MW) at bus I
             J=2   generation (MW) at bus  I
             J=3  max power flow (MW) allowed at bus I
    :param int NR: bus no. of the Ref bus

    :return: (*tuple*) -- a pair of numpy.ndarray objects, i.e.,
             BB: updated B matrix (admittance matrix for DC power flow)
             LT: array recording the original bus no.
    """

    LT = np.append(np.delete(BN[:, 0], NR), BN[NR, 0])
    BB = np.delete(np.delete(BB, obj=NR, axis=0), obj=NR, axis=1)

    return BB, LT
