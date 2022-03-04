def flo(LP, BLP, FLOW, THET, SFLOW):
    """
    Computes line flows

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

    :param numpy.ndarray FLOW: vector of line power flows (MW),  shape: (NLINES, )

    :param numpy.ndarray THET: bus (area) angle, shape: (NOAREA, )

    :param numpy.ndarray SFLOW: vector of bus (area) injection power (MW), shape: (NOAREA, )
    """

    for i in range(LP.shape[0]):
        j = LP[i, 1]  # from bus (area)
        k = LP[i, 2]  # to bus (area)
        FLOW[i] = (THET[j] - THET[k]) * BLP[i, 0]
        SFLOW[k] += FLOW[i]
        SFLOW[j] -= FLOW[i]
