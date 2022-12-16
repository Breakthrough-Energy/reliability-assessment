import numpy as np

from reliabilityassessment.monte_carlo.tm1 import tm1
from reliabilityassessment.monte_carlo.tm2 import tm2


def transm(
    IFLAG,
    JFLAG,
    CLOCK,
    JHOUR,
    IOI,
    BN,
    BLPA,
    BLP0,
    BB,
    LT,
    ZB,
    LP,
    NR,
    STMULT,
    LNSTAT,
    NLS,
    TRNSFR,
    TRNSFJ,
    SYSCON,
    CAPREQ,
    PLNDST,
    PCTAVL,
):

    """
    The main entry function of the transmsison module used in reliability assessment

    .. note:: 1) For descriptions of input and output variables, please refer to `variable
              descriptions.xlsx` in the project Dropbox folder at:
              https://www.dropbox.com/s/eahg8x584s9pg4j/variable%20descriptions.xlsx?dl=0
              2) arrays are modified in-place.
    """

    # local variables
    LCLOCK = CLOCK
    JHOUR = JHOUR
    IOI = IOI
    NOAREA = BN.shape[0]
    NN = NOAREA
    NLINES = LNSTAT.shape[0]
    NL = NLINES
    BLP = np.zeros((NLINES, 3))  # in original Fortran, np.zeros((100,3))

    if IFLAG != 1:
        for i in range(NL):
            BLP[i, 0] = BLPA[i, 0]
            BLP[i, 1] = BLPA[i, 1]
            BLP[i, 2] = BLPA[i, 2]
        # CALL TM1(BN,LP,BLP,BLP0,NN,NL,BB,ZB,LT,NR)
        BLP0[:], BB[:, :], LT[:], ZB[:, :] = tm1(BN, LP, BLP, NR)

    for i in range(NL):
        L = (LNSTAT[i] - 1) * 3  # LNSTAT[i] is 1-based value
        BLP[i, 0] = BLPA[i, L + 0]
        BLP[i, 1] = BLPA[i, L + 1] * STMULT[i, 0]
        BLP[i, 2] = BLPA[i, L + 2] * STMULT[i, 1]

    # local variable
    BNS = np.zeros((NOAREA, 6))  # in original Fortran, np.zeros((20,6))

    if JFLAG != 1:
        NLST = NLS
        NLS = 1
        for i in range(NN):
            BN[i, 2] = TRNSFR[i] + TRNSFJ[i]
            BN[i, 1] = 0.0
            BNS[i, 0] = BN[i, 2]
            BNS[i, 1] = BN[i, 1]
            BNS[i, 4] = SYSCON[i]
            BNS[i, 5] = CAPREQ[i]
            X = SYSCON[i] - CAPREQ[i]

            if X >= 0:
                if BN[i, 2] < 0:
                    BN[i, 2] = 0.0
            else:
                if X > BN[i, 2]:
                    BN[i, 2] = X

        FLOW, INDIC = tm2(
            BN,
            LP,
            BLP,
            BLP0,
            BB,
            ZB,
            LT,
            NR,
            NLS,
            BNS,
            LCLOCK,
            JHOUR,
            IOI,
            JFLAG,
            PLNDST,
            PCTAVL,
        )

        NLS = NLST
        KFLAG = 1

        SADJ = np.zeros(NN)
        for i in range(NN):
            SADJ[i] = SYSCON[i] - CAPREQ[i] - BN[i, 1]
            if SADJ[i] < 0:
                KFLAG = 0

        if KFLAG == 1:
            return JFLAG, FLOW, SADJ

        JFLAG = 1
        CADJ = np.zeros(NN)  # CADJ is locally used
        for i in range(NN):
            CADJ[i] = BN[i, 1] * NLS

    for i in range(NN):
        BN[i, 2] = SYSCON[i] - CAPREQ[i] - CADJ[i]
        BN[i, 1] = CAPREQ[i]
        BNS[i, 0] = BN[i, 2]
        BNS[i, 1] = BN[i, 1]
        BNS[i, 3] = BN[i, 3]

    if NLS != 0:
        for i in range(NL):
            j = LP[i, 1]
            K = LP[i, 2]
            BLP[i, 1] -= FLOW[i]
            BLP[i, 2] += FLOW[i]
            if BLP[i, 1] < 0.0:
                BLP[i, 1] = 0.0
            if BLP[i, 2] < 0.0:
                BLP[i, 2] = 0.0
            BN[j, 3] += FLOW[i]
            BN[j, 4] -= FLOW[i]
            BN[K, 3] -= FLOW[i]
            BN[K, 4] += FLOW[i]

    FLOW, INDIC = tm2(
        BN,
        LP,
        BLP,
        BLP0,
        BB,
        ZB,
        LT,
        NR,
        NLS,
        BNS,
        LCLOCK,
        JHOUR,
        IOI,
        JFLAG,
        PLNDST,
        PCTAVL,
    )

    for i in range(NN):
        BN[i, 3] = BNS[i, 3]
        BN[i, 4] = BNS[i, 3]  # a typo ?

    for i in range(NN):
        SADJ[i] = BN[i, 1] - SYSCON[i] + CAPREQ[i] + CADJ[i]

    return JFLAG, FLOW, SADJ
