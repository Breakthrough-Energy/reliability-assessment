def thetac(THET, THETC, LT, NN):
    """
    Generates vector of bus angles for all buses including ref bus

    :param numpy.ndarray THET: angle at every node of the ADM matrix
    :param numpy.ndarray THETC: angle of actual node corresponding to each node of the ADM matrix
    :param numpy.ndarray LT: actual node corresponding to node i of the ADM matrix
    :param int NN: number of nodes
    """
    for i in range(NN):
        THETC[i] = 0.0

    # subtract 1 as "reference" bus ALWAYS has a zero phase angle
    for i in range(NN - 1):
        j = LT[i]
        THETC[j] = THET[i]
