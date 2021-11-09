import numpy as np


def gsm(N, KT, KP, P1, P2, P3, PA, KA):
    """
    This subroutine adds a 2-state or a 3-state unit model to
    the generation capacity-outage model

    :param int N: (possibly) the total number of starting states (need to check later)
    :param float KT: rounded (to the nearest integer) fullc apacity of the given gen unit (to be added)
    :param float KP: rounded (to the nearest integer) derated capacity of the given gen unit (to be added)
    :param float P1: exact prob. of the I-th gen unit at its full capacity
    :param float P2: exact prob. of the I-th gen unit at its derated capacity
    :param float P3: exact prob. of the I-th gen unit in total loss (i.e. zero MW)
    :param numpy.ndarray PA: CUT OFF probability in the CAP outage table
    :param numpy.ndarray KA: CUT OFF capacity in the CAP outage table

    :return: (*int*) -- N (KA and PA are modified in-place)
    """
    INGS = 5000
    INGS1 = INGS - 50

    # Create outage states after unit addition
    KO = np.zeros((5000, 1))  # for local use only
    P = np.zeros((5000, 1))  # for local use only

    for i in range(N):
        KO[i, 0] = KA[i]
        KO[i, 1] = KA[i] + KP
        KO[i, 2] = KA[i] + KT
        P[i] = PA[i]

    NN = N + 1
    NNN = NN + 1

    for i in range(NN - 1, NNN):
        KO[i, 0] = 1000000
        KO[i, 1] = 1000000
        KO[i, 2] = 1000000
        P[i] = 0.0

    # Select outage states in ascending order of magnitude and calculate their probabilities and frequencies
    L = 1
    J1 = 2
    J2 = 1
    if KP == 0:
        J2 = J2 + 1
    J3 = 1
    KA[L] = 0
    PA[L] = 1.0

    while True:
        L = L + 1
        KCH = min(KO[J1, 0], KO[J2, 1], KO[J3, 2])
        KA[L] = KCH
        PA[L] = P[J1] * P1 + P[J2] * P2 + P[J3] * P3
        if KCH == KO[J1, 0]:
            J1 = J1 + 1
        if KCH == KO[J2, 1]:
            J2 = J2 + 1
        if KCH == KO[J3, 2]:
            J3 = J3 + 1
        JM = min(J1, J2, J3)

        if JM > N:
            break

        if PA[L] <= 0.1e-08:
            break

        if L > INGS1:
            break
    N = L

    if L > INGS1:
        print(
            "to prevent array overflow, CUT OFF PROB. in CAP outage table reduced to %.6f"
            % (PA[L])
        )

    NN = N + 1
    PA[NN] = 0.0

    return N
