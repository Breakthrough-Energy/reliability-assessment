import numpy as np


def resca(SUSTAT_1, MAXDAY, QTR, CAPOWN, RATES, JENT, INTCH):
    """
    Calculate the total owned cap
    (considering both “interchange at peak” and the seasonal effects)

    :param numpy.ndarray SUSTAT_1: total pure generation (MW) statistics of each area
                                   (initialzied by PKLOAD(J) and has multiple meanings;
                                    the meaning is subjected to change*)
                                   shape (NOAREA,)
                                   It is in fact the 1-th col of the 2D array 'SUSTAT' in the parent function

    :param int MAXDAY: the integer index of the day, then the daily-peak laod is mxiimum
                       of the whole year (365days)
                       shape (NOAREA,)

    :param numpy.ndarray QTR: quaterly index (in units of hours); shape (3,)
                              e.g. [13*168+0.5, 13*2*168+0.5, 13*3*168+0.5]

    :param numpy.array CAPOWN: shape (NOAREA, NUNITS), the (j,i) th element means
                               the fraction of unit i owned by area j.

    :param numpy.array RATES: shape (NUNITS, 4), (i,j) th element means
                              the rated power (MW) of unit i in the j-th season.

    :param numpy.ndarray JENT: 2D array of pointer identifying a specific contract,
                               shape (NOAREA, NOAREA),
                               the (i,j)-th element means the contract (no.) between area-i and area-j

    :param numpy.ndarray INTCH: 2D array of the contract content,
                                shape (total numebr of contracts, 365)
                                the (JENT[i,j], k)-th element means the contracted power (MW) on the kth day
                                from area-i to area-j
    """

    NOAREA = SUSTAT_1.shape[0]
    for i in range(NOAREA):
        IDAY = MAXDAY[i]
        IPER = np.argmax(QTR > IDAY * 24)
        NUINTS = CAPOWN.shape[1]
        for j in range(NUINTS):
            SUSTAT_1[i] += CAPOWN[i, j] * RATES[j, IPER]

        for j1 in range(NOAREA):
            if JENT[i, j1] != -1:
                IP = JENT[i, j1]
                SUSTAT_1[i] -= INTCH[IP, IDAY]

            if JENT[j1, i] != -1:
                IP = JENT[j1, i]
                SUSTAT_1[i] += INTCH[IP, IDAY]


def _resca(SUSTAT, MAXDAY, QTR, CAPOWN, RATES, JENT, INTCH):
    """
    Calculate the total owned cap
    (considering both “interchange at peak” and the seasonal effects)

    :param numpy.ndarray SUSTAT: statistics of each area
                                 shape (NOAREA,6)
                                 SUSTAT(J,1):  total pure load (MW) (initialzied by PKLOAD(J)
                                 SUSTAT(J,2):  total pure generation (MW) (initialzied by PKLOAD(J)
                                               (*has multiple meanings; its meaning is subjected to change*)
                                 SUSTAT(J,3):  (SUSTAT(I,2)-SUSTAT(I,1))/SUSTAT(I,1)*100.
                                                i.e. total pure generation (MW)
                                                (*has multiple meanings; its meaning is subjected to change*)
                                 SUSTAT(J,4): XLOL, loss of load (MW) (need to check later)
                                 SUSTAT(J,5): XLOLP, loss of load probability
                                 SUSTAT(J,6): EUE, expected unsereved enerrgy (MWh)

    :param int MAXDAY: the integer index of the day, then the daily-peak laod is mxiimum
                       of the whole year (365days)
                       shape (NOAREA,)

    :param numpy.ndarray QTR: quaterly index (in units of hours); shape (3,)
                              e.g. [13*168+0.5, 13*2*168+0.5, 13*3*168+0.5]

    :param numpy.array CAPOWN: shape (NOAREA, NUNITS), the (j,i) th element means
                               the fraction of unit i owned by area j.

    :param numpy.array RATES: shape (NUNITS, 4), (i,j) th element means
                              the rated power (MW) of unit i in the j-th season.

    :param numpy.ndarray JENT: 2D array of pointer identifying a specific contract,
                               shape (NOAREA, NOAREA),
                               the (i,j)-th element means the contract (no.) between area-i and area-j

    :param numpy.ndarray INTCH: 2D array of the contract content,
                                shape (total numebr of contracts, 365)
                                the (JENT[i,j], k)-th element means the contracted power (MW) on the kth day
                                from area-i to area-j
    """

    NOAREA = SUSTAT.shape[0]
    for i in range(NOAREA):
        IDAY = MAXDAY[i]
        IHR = IDAY * 24
        IPER = 0

        if IHR > QTR[0]:
            IPER = 1

        if IHR > QTR[1]:
            IPER = 2

        if IHR > QTR[2]:
            IPER = 3

        NUINTS = CAPOWN.shape[1]
        for j in range(NUINTS):
            SUSTAT[i, 1] += CAPOWN[i, j] * RATES[j, IPER]

        for j1 in range(NOAREA):
            if JENT[i, j1] != -1:
                IP = JENT[i, j1]
                SUSTAT[i, 1] -= INTCH[IP, IDAY]

            if JENT[j1, i] != -1:
                IP = JENT[j1, i]
                SUSTAT[i, 1] += INTCH[IP, IDAY]
