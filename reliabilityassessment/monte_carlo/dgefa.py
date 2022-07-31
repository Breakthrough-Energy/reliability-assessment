import numpy as np


def dgefa(A, LDA, N):
    """
    Factors a double precision matrix by gaussian elimination.

    dgefa is usually called by dgeco, but it can be called directly with
    a saving in time if rcond is not needed.
    (time for dgeco) = (1 + 9/n)*(time for dgefa).


    :param numpy.ndarray A: the matrix to be factored
    :param int LDA: the leading dimension of A
    :param int N: the order of the matrix  A

    :return: (*tuple*) -- INFO: an integer
                                = 0  normal value.
                                = K  if  U[K,K] == 0.0.  This is not an error condition
                                for this function, but it does indicate that dgedi will
                                divide by zero if called. Use RCOND in dgeco for a
                                reliable indication of singularity.
                          IPVT: an integer vector of pivot indices, shape (N,)

    .. note:: 1) arrays are modified in place.
              2) if RCOND is not needed, dgefa is slightly faster.
              3) on return, A  will be an upper triangular matrix and the multipliers
              which were used to obtain it. the factorization can be written  A = L*U,
              where L is a product of permutation and unit lower triangular matrices
              and  U  is upper triangular.
    """

    IPVT = np.zeros(N)

    #  Gaussian elimination with partial pivoting
    INFO = 0
    NM1 = N - 1
    if NM1 >= 1:
        for K in range(NM1):
            KP1 = K + 1

            # Find L = "pivot index"
            L = np.argmax(A[K:N, K]) + K
            IPVT[K] = L

            # zero pivot implies this column already triangularized
            if A[L, K] != 0.0:
                # INTERCHANGE IF NECESSARY
                if L != K:
                    A[L, K], A[K, K] = A[K, K], A[L, K]

                # COMPUTE MULTIPLIERS
                T = -1.0 / A[K, K]
                A[K + 1 : N, K] *= T  # DSCAL(N-K,T,A[K+1,K],1)

                # row elimination with column indexing
                for J in range(
                    KP1 - 1, N
                ):  # in original Fortran: from KP1 to N(included)
                    T = A[L, J]
                    if L != K:
                        A[L, J] = A[K, J]
                        A[K, J] = T
                    A[K + 1 : N, J] += (
                        T * A[K + 1 : N, K]
                    )  # DAXPY[N-K,T,A[K+1,K],1,A[K+1,J],1)
            else:
                INFO = K

    IPVT[N - 1] = N - 1  # in original Fortran: IPVT[N] = N
    if A[N, N] == 0.0:
        INFO = N

    return INFO, IPVT
