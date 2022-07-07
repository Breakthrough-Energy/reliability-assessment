import numpy as np

from reliabilityassessment.data_processing.pmsc import pmsc


def smaint(NOAREA, ID):
    """
    Obtain the (gen unit) maintenance schedule table ('JPLOUT') for each area

    :param int NOAREA: total number of areas
    :param numpy.ndarray ID:shape (NUNITS, 8)
                         ID(I,K), K=0: unit number (0-based)
                                  K=1: plant number (0-based)
                                  K=2: area of location (0-based)
                                  K=3: starting week of first planned outage (0-based)
                                  K=4: duration of first planned outage in weeks
                                  K=5: starting week of second planned outage (0-based)
                                  K=6: duration of second planned outage in weeks
                                  K=7: 1 if maintenance is pre-scheduled;
                                       0 if set automatically by the program
    :return: (*numpy.ndarray*) -- table of planned outages of units, JPLOUT, with
        shape (52, NUINTS+1), 1-dim: week index; 2-dim: the total number of gen units
        scheduled for maintenance in the corresponding week followed by a sequence
        of 0-based generator unit number

    .. note:: ID is modified in place
    """
    NUNITS = ID.shape[0]
    for areaIdx in range(NOAREA):
        # automatic scheduling of planned maintenance
        # for the specified area here; pmsc will update array ID
        pmsc(areaIdx, ID)

    JPLOUT = (-1) * np.ones((52, 1 + NUNITS), dtype=int)
    JPLOUT[:, 0] = 0

    for i in range(NUNITS):
        for wi in {3, 5}:
            if ID[i, wi] != -1:
                j1 = ID[i, wi]
                # the index of last week of a simulation year is 51
                j2 = min(51, j1 + ID[i, wi + 1])
                # increase total number of units under maintenance for the weeks
                JPLOUT[j1:j2, 0] += 1
                JPLOUT[np.arange(j1, j2), JPLOUT[j1:j2, 0]] = int(ID[i, 0])

    return JPLOUT
