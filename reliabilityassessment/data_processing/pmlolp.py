import numpy as np

from reliabilityassessment.data_processing.gsm import gsm


def pmlolp(CAPLOS, RD, NGU):
    """
    Computes the capacity outage probability table
    (for computing the 'M' factor used in function 'pmsc')

    :param numpy.ndarray CAPLOS: 1D array of the (assumed) capacity loss of each gen unit

    :param numpy.ndarray RD: 2D array of mixed information of each gen unit in a specific area
                         shape: (NUINTS, 3)
                         RD(I,0) // exact prob. of the I-th gen unit at its derated capacity
                         RD(I,1) // exact prob. of the I-th gen unit in total loss (i.e., zero MW)
                         RD(I,2) // the derated capacity (MW) of the I-th gen unit
                         I is the internal ordering of the specific area.

    :param int NGU: the total number of gen units in the specific area.

    :return: (*tuple*) -- a tuple of multiple arrays, i.e.,
                        NGS: int, the total number of generation states after adding a specific gen unit
                        KA: 1D array CUT-OFF capacity in the CAP-outage-table
                        PA: 1D array CUT-OFF probability in the CAP-outage-table
    """

    INC = 10
    XINC = 5.6
    CAPIND = 0.0

    # (INGS) 5000 : possible maximum total number of system generation states
    KA = np.zeros((5000))
    PA = np.zeros((5000))

    PA[0] = 1.0
    NGS = 1

    # add units to capacity outage probability table
    for i in range(NGU):
        NT = i
        # update the system capacity up to this unit
        CAPIND = CAPIND + CAPLOS[NT]
        KT = CAPLOS[NT]
        TK = KT
        KT = (TK + XINC - 0.1) / INC
        KT = KT * INC
        PK = KT

        # compute the derating and round off to 'INC''
        PK = PK * RD[NT, 2]
        KP = (PK + XINC - 0.1) / INC
        KP = KP * INC

        P2 = RD[NT, 0]
        P3 = RD[NT, 1]
        P1 = 1 - P2 - P3
        IRAT = P1 * 10**6

        if IRAT != 0:
            # call ‘gsm’ for unit addition to the generation system model
            # KA and PA are modified by 'gsm' in-place
            NGS = gsm(NGS, KT, KP, P1, P2, P3, PA, KA)

    # return when all units have been added
    return NGS, KA, PA
