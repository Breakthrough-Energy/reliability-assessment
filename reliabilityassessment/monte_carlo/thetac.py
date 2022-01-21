import numpy as np

def thetac(THET, THETC, LT, NN):
    """
    Generates vector of bus angles for all buses including ref bus

    :param numpy.ndarray THET: angle at every node of the ADM matrix
    :param numpy.ndarray THETC: angle of actual node corresponding to each node of the ADM matrix
    :param numpy.ndarray LT: actual node corresponding to node i of the ADM matrix
    :param int NN: number of nodes
    """
    THETC = np.zeros(NN)

    # last array value is the "reference" bus that ALWAYS has a zero phase angle
    THETC[LT[:-2]] = THET[:-2]