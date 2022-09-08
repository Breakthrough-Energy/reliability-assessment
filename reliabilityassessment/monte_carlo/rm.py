import numpy as np


def rm(M, CB, TAB):
    """
    2D Matrix left-multiplication by a vector within a specified range

    :param int M: the specified range for the matrix (to be multiplied)
    :param numpy.ndarray CB: the input vector (as the be left-multiplier)
                             its length could be larger than M.
    :param numpy.ndarray TAB: the input matrix (to be multiplied)
                             its row/col size could be larger than M.
    :return: (*numpy.ndarray*) -- PY: 1D array with shape (M,)
    """

    PY = np.zeros(M)
    for i in range(M):
        for j in range(M):
            PY[i] += CB[j] * TAB[j, i]
    return PY
