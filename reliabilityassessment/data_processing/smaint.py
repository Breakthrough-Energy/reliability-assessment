import numpy as np

from reliabilityassessment.data_processing import pmsc


def smaint(NOAREA, ID):
    """
    Obtain the (gen unit) maintenance schedule table ('JPLOUT') for each area

    :param int NOAREA: total number of areas

    :param numpy.ndarray ID:shape (NUNITS, 8)
                    ID(I,K),K=0 to 2 : unit number (1-based), plant number (1-based), area of location (str)
                            K=3: starting week of first planned outage (1-based index)
                            K=4: duration of first planned outage in weeks
                            K=5: starting week of second planned outage
                            K=6: duration of second planned outage
                            K=7: 1 if maintenance is prescheduled, and 0 if
                            set automatically by the program.

    :return: (*numpy.ndarray*) -- JPLOUT - table of planned outages of units.
                             shape (52, NUINTS+1) (need to check later).
                             1-dim: week index;
                             2-dim: 1st-element：the total count of gen units scheduled for
                                                 maintenance in a specified week.
                                    other elements: gen unit no. (0-based for Python computation)
                             And implicitly update array 'ID'
    """
    NUNITS = ID.shape[0]
    for areaIdx in range(NOAREA):
        # automatic scheduling of planned maintenance
        # for the specified area here; pmsc will update array ID
        pmsc(areaIdx, ID)

    JPLOUT = np.zeros((52, NUNITS))  # in Fortran, np.zeros((52, 120))

    for i in range(NUNITS):
        if ID[i, 3] != 0:
            j1 = ID[i, 3] - 1  # convert to 0-based index
            j2 = j1 + ID[i, 4] - 1
            if j2 > 51:  # 52-1
                j2 = 51
            for j in range(j1, j2 + 1):
                JPLOUT[
                    j, 0
                ] += 1  # increase the total count of gen units to be maintained
                JPLOUT[j, JPLOUT[j, 0]] = (
                    ID[i, 0] - 1
                )  # convert to 0-based index gen unit idx

        if ID[i, 5] != 0:
            j1 = ID[i, 5] - 1  # convert to 0-based index
            j2 = j1 + ID[i, 6] - 1
            if j2 > 51:  # 52-1
                j2 = 51
            for j in range(j1, j2 + 1):
                JPLOUT[
                    j, 0
                ] += 1  # increase the total count of gen units to be maintained
                JPLOUT[j, JPLOUT[j, 0]] = (
                    ID[i, 0] - 1
                )  # convert to 0-based index gen unit idx

    return JPLOUT
