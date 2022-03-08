import numpy as np


def flo(LP, BLP, THET):
    """
    Computes line power flows and bus (power) injection

    :param numpy.ndarray LP: array storing line related information
                            shape: (NLINES, 3)
                            LP(I,J)  I  ENTRY  NUMBER
                            LP(I,0)  LINE NUMBER SET=I
                            LP(I,1)  STARTING NODE
                            LP(I,2)  ENDING NODE

    :param numpy.ndarray BLP: contains the data on admittance, cap and backward cap
                              of the line at position I in LP(I,J) (at the state-1)
                              shape: (NLINES, 3)
                              BLP(I,J) I  ENTRY  NUMBER
                              BLP(I,0)  admittance
                              BLP(I,1)  capacity (MW)
                              BLP(I,2)  backward capacity (MW)

    :param numpy.ndarray THET: bus (area) angle, shape: (NOAREA, )

    :return: (*tuple*)  -- SFLOW: vector of bus (area) injection power (MW)
                                        shape (NOAREA, )
                           FLOW: vector of line power flows (MW),  shape: (NLINES, )
    """

    NLINES = LP.shape[0]
    NOAREA = len(THET)
    SFLOW = np.zeros((NOAREA,))
    FLOW = np.zeros((NLINES,))

    for i in range(NLINES):
        j = LP[i, 1]  # from bus (area)
        k = LP[i, 2]  # to bus (area)
        FLOW[i] = (THET[j] - THET[k]) * BLP[i, 0]
        SFLOW[k] += FLOW[i]
        SFLOW[j] -= FLOW[i]

    return SFLOW, FLOW
