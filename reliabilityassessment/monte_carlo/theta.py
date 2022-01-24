import numpy as np


def theta(Z, INJ):
    """
    Computes node angles for nodes of ZBus impedance matrix. This function is used during step 7 in
    the General Program Logic section of the project README.

    :param numpy.ndarray Z: Zbus impedence matrix
    :param numpy.ndarray INJ:  bus injection vector

    :return: (*numpy.ndarray*) -- THET: node angles vector
    """
    THET = np.dot(Z, INJ)
    return THET
