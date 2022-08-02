import numpy as np

from reliabilityassessment.monte_carlo.filem import filem
from reliabilityassessment.monte_carlo.findld import findld
from reliabilityassessment.monte_carlo.findtn import findtn
from reliabilityassessment.monte_carlo.gstate import gstate
from reliabilityassessment.monte_carlo.hrstat import hrstat
from reliabilityassessment.monte_carlo.lstate import lstate
from reliabilityassessment.monte_carlo.pkstat import pkstat
from reliabilityassessment.monte_carlo.sumcap import sumcap
from reliabilityassessment.monte_carlo.transm import transm


def hour(
    ATRIB,
    CLOCK,
    MFA,
    JSTEP,
    JFREQ,
    NUMINQ,
    IPOINT,
    EVNTS,
    MXPLHR,
    PROBG,
    RATING,
    DERATE,
    PLNDST,
    CAPOWN,
    CAPCON,
    PROBL,
    JENT,
    INTCH,
    MXCRIT,
    JCRIT,
    NFCST,
    HRLOAD,
    FCTERR,
    LSFLG,
    LOLTHP,
    LOLGHP,
    LOLTHA,
    MGNTHA,
    MGNTHP,
    LOLGHA,
    MGNGHA,
    MGNGHP,
    LOLTPP,
    LOLGPP,
    LOLTPA,
    MGNTPA,
    MGNTPP,
    LOLGPA,
    MGNGPA,
    MGNGPP,
    LINENO,
    LP,
    BLPA,
    IOI,
    BN,
    NR,
    NLS,
):

    """
    Schedule hourly simulation events

    :return: (*tuple*)
    .. note:: 1) For descriptions of input and output variables, please refer to `variable
              descriptions.xlsx` in the project Dropbox folder at:
              https://www.dropbox.com/s/eahg8x584s9pg4j/variable%20descriptions.xlsx?dl=0
              2) arrays are modified in-place.

    .. note:: EVNTS is modified in place
    """

    IFLAG, JFLAG = 0, 0

    ATRIB[0] = CLOCK + JSTEP

    # may need modify here: some retun var can be implciit
    EVNTS, NUMINQ, MFA, IPOINT = filem(MFA, ATRIB, NUMINQ, IPOINT, EVNTS)

    print("Entering subroutine hour")

    # ..............UPDATE HOUR NUMBER.............
    DXX = 8760
    JHOUR = CLOCK % DXX  # DMOD(CLOCK,DXX)
    # if JHOUR == 0: # original Fortran, since 1-based
    #     JHOUR=8760 # original Fortran
    JJ = JHOUR % JFREQ

    # need to double check
    JDAY = np.floor(JHOUR / 24)  # original Fortran, (JHOUR-.1)/24 + 1, since 1-based

    # if JDAY > 365: original Fortran, commented by Yongli no need in Python
    #     JDAY = 365 original Fortran, commented by Yongli no need in Python

    JHOURT = JHOUR
    if JSTEP == 24:  # if only aily peak stats  are collected
        JHOUR = MXPLHR[JDAY]  # MXPLHR stores the pool peak hour

    # ......Draw generator states and sum capacities by area...............
    if JJ == 0:  # this logic might make JFREQ redundant;
        JJ = 1  # will check it with the original authors

    NOAREA = CAPOWN.shape[0]
    NUNITS = CAPOWN.shape[1]

    if JJ == 1.0 or JFREQ == 1:
        PCTAVL, AVAIL = gstate(
            PROBG, RATING, DERATE, PLNDST, rng=np.random.default_rng()
        )
        CAPAVL, SYSOWN, SYSCON, TRNSFJ = sumcap(NUNITS, NOAREA, AVAIL, CAPOWN, CAPCON)
        LNSTAT = lstate(PROBL, test_seed=None)
        # CALLS = 0. # a flag variable that will be used later
        TRNSFR = findtn(NOAREA, JENT, INTCH, JDAY)

    # STMULT(I,1) IS FORWARD DIRECTION ADJUSTMENT
    # STMULT(I,2) IS BACKWARD DIRECTION ADJUSTMENT
    NLINES = len(LNSTAT)
    STMULT = np.zeros((NLINES, 2))
    for i in range(NLINES):
        for j in range(2):
            STMULT[i, j] = 1.0

    JPNT = 0  # in original Fortran, 1

    for i in range(MXCRIT):
        if JCRIT[JPNT] == 0:
            JPNT = JPNT + 1
        else:
            NOCRIT = JCRIT[JPNT + 1]
            CRTVAL = 0.0
            for ii in range(NOCRIT):
                if JCRIT[JPNT + 1 + ii] > 0 and JCRIT[JPNT + 1 + ii] <= 500 - 1:
                    # maybe 500 - 1 , need to check back later
                    CRTVAL += AVAIL[JCRIT[JPNT + 1 + ii]]
                else:
                    # the follwing commented logic are from Fortran, and looks meaning less
                    # if IGENE == 0:
                    #     print("Logic error...")
                    # IGENE=1
                    # I simplify and rephrase as follows:
                    print("\n Logic error happens regarding the JCRIT array ...\n")

            if CRTVAL == 0.0:
                STMULT[JCRIT[JPNT], 0] = float(JCRIT[JPNT + NOCRIT + 3]) / 100.0
                STMULT[JCRIT[JPNT], 1] = float(JCRIT[JPNT + NOCRIT + 4]) / 100.0

            JPNT += NOCRIT + 5

    for NST in range(NFCST):

        CAPREQ = findld(NST, JHOUR, HRLOAD, FCTERR)

        FLAG = 0.0
        # FLAG2 = 0.0  defined but not used in the original Fortran code
        MARGIN = np.zeros(NOAREA)

        for j in range(NOAREA):
            MARGIN[j] = int(SYSCON[j] - CAPREQ[j])  # doublec ehck  IFIX usage !
            #  IFIX(SYSCON[j] - CAPREQ[j])
            JMJ = int(SYSOWN[j] - CAPREQ[j])  # IFIX(SYSOWN[j] - CAPREQ[j])
            if JMJ < 0:
                pass  # FLAG2 = 1. defiend but not used in the original Fortran code
            if MARGIN[j] < 0:
                FLAG += 1.0
                IT = j

        if FLAG == 0.0:
            # print('Returning: No Loss')
            # might be no need to return JHOUR, JDAY; will check later
            return (
                NUMINQ,
                MFA,
                IPOINT,
                JHOURT,
                JHOUR,
                JDAY,
                PCTAVL,
                AVAIL,
                CAPAVL,
                SYSOWN,
                SYSCON,
                np.zeros((NOAREA, NOAREA)),
                np.zeros(NOAREA),
            )

        NSHRT = int(FLAG)  # IFIX(FLAG)
        LNCAP = np.zeros(NLINES)

        if NSHRT == NOAREA:
            hrstat(
                NST,
                MARGIN,
                LSFLG,
                LOLTHP,
                LOLGHP,
                LOLTHA,
                MGNTHA,
                MGNTHP,
                LOLGHA,
                MGNGHA,
                MGNGHP,
            )
            if JHOUR == MXPLHR[JDAY]:
                pkstat(
                    NST,
                    MARGIN,
                    LOLTPP,
                    LOLGPP,
                    LOLTPA,
                    MGNTPA,
                    MGNTPP,
                    LOLGPA,
                    MGNGPA,
                    MGNGPP,
                )
        else:
            if FLAG > 1.0:
                pass  # as the original Fortran logic
            else:
                NEED = abs(MARGIN[IT])
                for j in range(NOAREA):
                    if MARGIN[j] > 0:
                        N = LINENO[j, IT]
                        if N > 0:
                            M = (LNSTAT[N] - 1) * 3 + 1  # in original Fortran, +2
                            if LP[N, 1] == j:
                                XX = BLPA[N, M]
                                LNCAP[N] = int(XX)
                            else:
                                XX = BLPA[N, M + 1]
                                LNCAP[N] = int(XX)
                            MXHELP = min(MARGIN[j], LNCAP[N])
                            NEED = NEED - MXHELP

            for j in range(NOAREA):
                if MARGIN[j] < 0:
                    TEMPFL = 1.0

            JFLAG = 0

            # BLP0, FLOW might be removed from return values of 'transm'
            # will check later
            BLP0, BB, LT, ZB, FLOW, SADJ = transm(
                IFLAG,
                JFLAG,
                CLOCK,
                JHOUR,
                IOI,
                BN,
                BLPA,
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
            )

            # CALLS = 1.
            if IFLAG == 0:
                IFLAG = 1
            if JFLAG == 0:
                # might be no need to return JHOUR, JDAY; will check later
                return (
                    NUMINQ,
                    MFA,
                    IPOINT,
                    JHOURT,
                    JHOUR,
                    JDAY,
                    PCTAVL,
                    AVAIL,
                    CAPAVL,
                    SYSOWN,
                    SYSCON,
                    BB,
                    LT,
                    ZB,
                )
            TEMPFL = 0.0

            for j in range(NOAREA):
                ZZ = -SADJ[j]
                MARGIN[j] = int(ZZ)  # IFIX(ZZ)
                if MARGIN[j] < 0:
                    TEMPFL = 1.0

            if TEMPFL == 0:
                # might be no need to return JHOUR, JDAY; will check later
                return (
                    NUMINQ,
                    MFA,
                    IPOINT,
                    JHOURT,
                    JHOUR,
                    JDAY,
                    PCTAVL,
                    AVAIL,
                    CAPAVL,
                    SYSOWN,
                    SYSCON,
                    BB,
                    LT,
                    ZB,
                )

    # might be no need to return JHOUR, JDAY; will check later
    return (
        NUMINQ,
        MFA,
        IPOINT,
        JHOURT,
        JHOUR,
        JDAY,
        PCTAVL,
        AVAIL,
        CAPAVL,
        SYSOWN,
        SYSCON,
        BB,
        LT,
        ZB,
    )
