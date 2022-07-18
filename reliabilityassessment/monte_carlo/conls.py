import numpy as np


def conls(
    A,
    BC,
    INJ,
    NX,
    XOB,
    IBAS,
    NR,
    LT,
    B,
    BLP,
    LP,
    NL,
    M,
    N,
    BN,
    B1,
    LOD,
    NLS,
    XOBI,
    BS,
    TAB,
    N1,
):
    """
     Update the coefficient-matrcies of constraints, and other quantities, for the LP-based DCOPF
     (load shedding) under the "load-sharing" mode (in short, "LS")

     :param numpy.ndarray A: initial shape (200, 250)
                             the realistic size is determined on-the-fly
                             a matrix used in LP-based DCOPF
                             (possibly) means the  coefficient matrix of all constraints in LP
     :param numpy.ndarray BC: initial shape (20, 20) (possibly NOAREA-by-NOAREA)
                             the realistic size is determined on-the-fly
                             a matrix used in LP-based DCOPF
     :param numpy.ndarray INJ: initial shape (20, ) (possibly NOAREA-by-1)
                             the realistic size is determined on-the-fly
                             a vector used in LP-based DCOPF
                             (possibly) related to the injection at each bus(i.e., "area")
     :param int NX: an integer used in LP-based DCOPF
                    (possibly) related to the variables/constraint dimensions in LP-based DCOPF
     :param numpy.ndarray XOB: initial shape (250, )
                             the realistic size is determined on-the-fly
                             a vector used in linear programming(LP)-based DCOPF
                             values are either 0 or 1; thus, possibly integer indicators
     :param numpy.ndarray IBAS: 1D array with initial shape (250, )
                             the realistic size is determined on-the-fly
                             a helper vector used in linear programming-based DCOPF
                             (possibly) means the indices of the "basis-vector" used in LP
     :param int NR: an integer used in LP-based DCOPF
                    (possibly) means the index of the reference bus (area) in power flow  computation
     :param numpy.ndarray LT: 1D array with initial shape (250, )
                               LT[I] contains the actual node index corresponding to node I
                               of the reduced admittance matrix here;  shape (NOAREA,)"
     :param numpy.ndarray B: 1D array with initial shape (200, )
                             the realistic size is determined on-the-fly
                             a vector used in linear programming(LP)-based DCOPF
     :param numpy.ndarray BLP: 2D array with shape (NLINES, 3)
                               BLP[I, 0]: admittance of the line at the I-th entry of LP
                               BLP[I, 1]: capacity (MW)
                               BLP[I, 2]: backward capacity (MW)
    :param numpy.ndarray LP: 2D array with shape (NLINES, 3)
                              LP[I, 0]: line number, set to I
                              LP[I, 1]: starting node
                              LP[I, 2]: ending node
     :param int NL: an integer used in LP-based DCOPF
                    means the number of lines
     :param int M: used in linear programming(LP)-based DCOPF
                   (possibly) means the total number of constraints
     :param int N: used in linear programming(LP)-based DCOPF
                   (possibly) means the total number of "decision variables" in the LP
     :param numpy.ndarray BN: 2D array with shape (NOAREA, 4)
                              BN[I, 0]: area (bus/node) number
                              BN[I, 1]: area load, overridden by net injection after
                                        calling :py:func: `tm2`
                              BN[I, 2]: area net injection, overridden by load
                                        curtailment after calling :py:func: `tm2`
                              BN[I, 3]: area constraint on total power flow
     :param numpy.ndarray B1: 1D array with initial shape (200, )
                             the realistic size is determined on-the-fly
                             a helper vector used in linear programming-based DCOPF
     :param numpy.ndarray LOD: 1D array with shape (NLINES, 3)
                               vector of loads
     :param int NLS: an indicator of loss sharing mode,
                     0    LOSS SHARING
                     1    NO LOSS SHARING
     :param numpy.ndarray XOBI: initial shape (2, 250)
                             the realistic size is determined on-the-fly
                             a matrix used in linear programming(LP)-based DCOPF
                             values are either 0 or 1; thus possibly integer indicators
     :param numpy.ndarray BS: 1D array with initial shape (200, )
                             the realistic size is determined on-the-fly
                             a helper vector used in linear programming-based DCOPF
     :param numpy.ndarray TAB: initial shape (200, 250)
                             the realistic size is determined on-the-fly
                             a matrix used in linear programming(LP)-based DCOPF
     :param int N1: an integer used in LP-based DCOPF

     :return: (*tuple*) -- a tuple of three scalars, M, N, N1;
                           and modify several arrays in place.
    """

    NX1 = NX + 1
    NP1 = 3 * NX + 1
    NP = 2 * NX + 1
    NP2 = NP1 + NX1
    NMAX1 = 250
    NMAX2 = 200

    IEQ = np.zeros(NX)  # originally in Fortran np.zeros(20)

    for i in range(NMAX1):
        IBAS[i] = 0.0
        XOB[i] = 0.0
        XOBI[0, i] = 0.0
        XOBI[1, i] = 0.0
        for j in range(NMAX2):
            A[j, i] = 0.0

    for i in range(NX):
        II = i
        j = LT[i]
        IEQ[j] = i

    IEQ[NR] = II + 1
    NXD = 2 * NX
    for i in range(NX):
        JJ = 1
        for j in range(NX):
            A[i, JJ] = BC[i, j]
            A[i, JJ + 1] = -BC[i, j]
            JJ = JJ + 2
        A[i, NXD + i] = -1.0
        j = LT[i]
        B[i] = -LOD[j]
        A[i, NP1 + i] = -1.0

    NX1 = NX + 1
    for i in range(NL):
        if LP[i, 2] == NR:
            K = LP[i, 1]
            J1 = IEQ[K]
            I1 = J1 * 2 - 1
            A[NX1, I1] = A[NX1, I1] + BLP[i, 0]
            A[NX1, I1 + 1] = A[NX1, I1 + 1] - BLP[i, 0]
        if LP[i, 1] == NR:
            K = LP[i, 2]
            J1 = IEQ[K]
            I1 = J1 * 2 - 1
            A[NX1, I1] = A[NX1, I1] + BLP[i, 0]
            A[NX1, I1 + 1] = A[NX1, I1 + 1] - BLP[i, 0]

    A[NX1, NX + NP] = 1
    B[NX1] = -LOD[NR]
    A[NX1, NP1 + NX1] = 1.0
    for i in range(NP2):
        A[NX1, i] = -A[NX1, i]

    NXD = 2 * NX
    for i in range(NX1):
        XOB[NXD + i] = 1.0
    for i in range(NX1):
        A[i + NX1, NP2 + i] = 1
        A[i + NX1, NXD + i] = 1
        j = LT[i]
        B[i + NX1] = LOD[j]

    NX2 = NX1 + NX1
    NP2 = 2 * NX + 2 * NX1
    for i in range(NX1):
        A[i + NX2, NP1 + i] = 1.0
        A[i + NX2, NP2 + NX1 + i] = 1.0
        j = LT[i]
        B[i + NX2] = LOD[j] + INJ[i]

    NX3 = 3 * NX1
    NP22 = NP2 + 2 * NX1
    NP3 = NP22 + NL

    for i in range(NL):
        I1 = NX3 + i
        I2 = I1 + NL
        J1 = LP[i, 1]
        J2 = LP[i, 2]
        J4 = IEQ[J1]
        J5 = IEQ[J2]
        J42 = J4 * 2
        J41 = J42 - 1
        J52 = J5 * 2
        J51 = J52 - 1
        if J4 <= NX:
            A[I1, J41] = BLP[i, 0]
            A[I1, J42] = -BLP[i, 0]
            A[I2, J41] = -BLP[i, 0]
            A[I2, J42] = BLP[i, 0]
        if J5 <= NX:
            A[I1, J51] = -BLP[i, 0]
            A[I1, J52] = BLP[i, 0]
            A[I2, J51] = BLP[i, 0]
            A[I2, J52] = -BLP[i, 0]
        A[I1, NP22 + i] = 1
        A[I2, NP3 + i] = 1
        B[I1] = BLP[i, 1]
        B[I2] = BLP[i, 2]

    NXL = NP3 + NL
    NXD1 = NXD + 1
    M = 3 * NX1 + 2 * NL
    N = NXL

    for i in range(NX1):
        for j in range(NX1):
            K = LT[i]
            A[i + M, j] = A[i, j]
            B[i + M] = BN[K, 3]
            A[i + M + NX1, j] = -A[i, j]
            B[i + M + NX1] = BN[K, 4]
        A[i + M, i + N] = 1.0
        A[i + M + NX1, i + N + NX1] = 1.0

    M += 2 * NX1
    N += 2 * NX1

    for i in range(M):
        B1[i] = B[i]
    for i in range(NX1):
        if B[i] > 0:
            B[i] = 0.0

    for i in range(M):
        if B[i] >= 0:
            continue
        for j in range(N):
            A[i, j] = -A[i, j]
        B[i] = -B[i]

    for i in range(NX1):
        A[i, i + N] = 1.0
        XOBI[0, i + N] = 1.0

    for i in range(N):
        XOBI[1, i] = XOB[i]

    N += NX1
    NXD2 = NXD1 + NX1 + NX1
    II = 0

    for i in range(NXD2 - 1, N):
        II += 1
        IBAS[II] = i

    NX2 = NX1 + 1
    j = 0
    for i in range(NX2 - 1, M):
        j = j + 1
        for K in range(N):
            TAB[j, K] = A[i, K]
        BS[j] = B[i]

    for i in range(NX1):
        j += 1
        for K in range(N):
            TAB[j, K] = A[i, K]
        BS[j] = B[i]

    for i in range(M):
        for j in range(N):
            A[i, j] = TAB[i, j]

    N1 = N - NX1

    return M, N, N1
