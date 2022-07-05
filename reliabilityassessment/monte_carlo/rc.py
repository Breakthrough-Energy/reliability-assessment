def rc(M, PY, A, IND):
    """
    A vector left-multiply a specified col of a matrix

    :param int M: the specified range for the matrix (to be multiplied)
    :param numpy.ndarray PY: the input 1D vector with shape (M,)
    :param numpy.ndarray A: the input matrix (to be multiplied)
                            its row/col size can be larger than M.
    :param int IND: the specified index of column of matrix A

    :return: (*float*) -- PROD: the product result
    """

    PROD = 0
    for i in range(M):
        PROD += PY[i] * A[i, IND]
    return PROD
