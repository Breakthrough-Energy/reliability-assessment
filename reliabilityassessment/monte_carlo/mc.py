import numpy as np


def mc(M, IN, TAB, A):
    """
    A vector right-multiply a matrix. The vector is a specified column of another matrix.

    :param int M: the specified range for the matrix (to be multiplied)
    :param int IN: the specified index of column of matrix A
    :param numpy.ndarray TAB: the input matrix (to be multiplied)
                              its row/col size could be larger than M.
    :param numpy.ndarray A: the input matrix (to be extracted one specific column)
                            its row/col size could be larger than M.
    :return: (*numpy.ndarray*) -- CO: the result vector
    """
    CO = np.zeros(M)
    for i in range(M):
        for j in range(M):
            CO[i] += TAB[i, j] * A[j, IN]
    return CO
