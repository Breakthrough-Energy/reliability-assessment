import numpy as np

from reliabilityassessment.monte_carlo.dgeco import dgeco
from reliabilityassessment.monte_carlo.dgedi import dgedi
from reliabilityassessment.monte_carlo.mc import mc
from reliabilityassessment.monte_carlo.rc import rc
from reliabilityassessment.monte_carlo.rm import rm


def linp(M, N, A, XOB, XOBI, IBAS, BS, TAB, LCLOCK, N1):
    """
    A customized version of linear programming (LP).
    (please note its inptus, outputs and internal logic may not be exaclty the same as
     that of the standard LP APIs in SciPy/NumPy or MATLAB)

    :param int M: used in linear programming(LP)-based DCOPF
                   (possibly) means the total number of constraints
    :param int N: used in linear programming(LP)-based DCOPF
                   (possibly) means the total number of "decision variables" in the LP
    :param numpy.ndarray A: initial shape (200, 250)
                             the realistic size is determined on-the-fly
                             a matrix used in LP-based DCOPF
                             (possibly) means the  coefficient matrix of all constraints in LP
    :param numpy.ndarray XOB: initial shape (250, )
                             the realistic size is determined on-the-fly
                             a vector used in linear programming(LP)-based DCOPF
                             values are either 0 or 1; thus, possibly integer indicators
    :param numpy.ndarray XOBI: initial shape (2, 250)
                             the realistic size is determined on-the-fly
                             a matrix used in linear programming(LP)-based DCOPF
                             values are either 0 or 1; thus possibly integer indicators
    :param numpy.ndarray IBAS: 1D array with initial shape (250, )
                             the realistic size is determined on-the-fly
                             a helper vector used in linear programming-based DCOPF
                             (possibly) means the indices of the "basis-vector" used in LP
    :param numpy.ndarray BS: 1D array with initial shape (200, )
                             the realistic size is determined on-the-fly
                             a helper vector used in linear programming-based DCOPF
    :param numpy.ndarray TAB: initial shape (200, 250)
                             the realistic size is determined on-the-fly
                             a matrix used in linear programming(LP)-based DCOPF
    :param float LCLOCK: the simulation time clock
    :param int N1: an integer used in LP-based DCOPF

    :return: (*int*) N -- see above definition.

    .. note:: arrays are modified in place.
    """

    BS1 = np.zeros(200)
    IBASF = np.zeros(250, dtype=int)
    CB = np.zeros(200)
    TABT = np.zeros(200, 250)

    for i in range(M):
        BS1[i] = BS[i]

    IR = 0
    INV = 0
    for i in range(N):
        XOB[i] = XOBI[0, i]
        IBASF[i] = 0

    IPH = 1
    for i in range(200):
        for j in range(250):
            TAB[i, j] = 0

    for i in range(M):
        k = IBAS[i]
        IBASF[k] = 1
        for j in range(M):
            TAB[i, j] = 0.0
        TAB[i, i] = 1
        CB[i] = XOB[k]

    while True:  # 110  while loop
        N = N1
        PY = rm(M, CB, TAB)

        gotoFlag1000 = False
        while True:  # 111 while loop
            CM = 0.0
            for i in range(N):
                if IBASF[i] == 1:
                    continue

                PROD = rc(M, PY, A, i)

                CC = XOB[i] - PROD
                CABA = abs(CC)
                if CABA < 0.1e-6:
                    CC = 0.0
                DIFF = CC - CM
                DIFF = abs(DIFF)
                if DIFF < 0.1e-06:
                    continue
                if CC < CM:
                    IND = i
                if CC < CM:
                    CM = CC

            if CM < 0:
                break  # GO TO 400

            if IR == 1:
                gotoFlag1000 = True
                break  # GO TO 1000

            for i in range(M):
                IMAX = 0
                K = IBAS[i]
                for j in range(M):
                    TABT[j, i] = A[j, K]

            IPVT = dgeco(TABT, 200, M)
            dgedi(TABT, 200, M, IPVT, 2)

            PY = rm(M, CB, TABT)

            for i in range(N):
                if IBASF[i] == 0:
                    continue
                PROD = rc(M, PY, A, i)
                CC = XOB[i] - PROD
                CC = abs(CC)
                if CC < 1e-8:
                    continue
                IMAX = 500

            if IMAX != 0:
                f = open("extra.txt", "w")
                f.write("\n     INVERSIONS = %2d LCLOCK = %6d\n" % (INV, LCLOCK))
                f.close()

            if IMAX == 500:
                INV += 1

            for i in range(M):
                BS[i] = 0.0
                for j in range(M):
                    BS[i] = BS[i] + TABT[i, j] * BS1[j]
                    TAB[i, j] = TABT[i, j]

            IR = 1
            # GO TO 111 while loop

        # 400 CONTINUE
        if gotoFlag1000 is False:
            IR = 0
            P = mc(M, IND, TAB, A)
            ICOUN = 0

            for i in range(M):
                if P[i] <= 0.1e-06:
                    continue
                BNUM = BS[i]
                BAB = abs(BNUM)
                if BAB < 0.1e-12:
                    BNUM = 0.0
                RAT = BNUM / P[i]
                ICOUN = ICOUN + 1

                if ICOUN == 1:
                    RAT1 = RAT
                    IPIV = i
                    continue
                if RAT < RAT1:
                    IPIV = i
                    RAT1 = RAT
                    continue
                if IPH != 2 and BS[i] == 0:
                    K = IBAS[i]
                    if XOB[K] == 1:
                        IPIV = i

            for i in range(M):
                if P[i] == 0:
                    continue
                if i == IPIV:
                    continue
                for j in range(M):
                    TAB[i, j] = TAB[i, j] - TAB[IPIV, j] * P[i] / P[IPIV]
                BS[i] = BS[i] - BS[IPIV] * P[i] / P[IPIV]

            for i in range(M):
                TAB[IPIV, i] = TAB[IPIV, i] / P[IPIV]

            BS[IPIV] = BS[IPIV] / P[IPIV]
            CB[IPIV] = XOB[IND]
            K1 = IBAS[IPIV]
            IBASF[K1] = 0
            IBASF[IND] = 1
            IBAS[IPIV] = IND
            SUMO = 0.0

            for i in range(M):
                K = IBAS[i]
                SUMO += BS[i] * XOB[K]
            # GO TO 110 while loop
        else:  # 1000  CONTINUE
            if IPH == 2:
                break  # GO TO 2000 (exit)
            # N -= NXT # this comment exists in the original Fortran code.
            for i in range(N):
                XOB[i] = XOBI[2, i]
            for i in range(N):
                K = IBAS[i]
                CB[i] = XOB[K]
            IPH = 2
        # GO TO 110 while loop

    # 2000 (exit)
    return N
