import numpy as np


def dgedi(A, LDA, N, IPVT, JOB):
    """
    Computes the determinant and inverse of a matrix
    using the factors computed by dgeco

    :param numpy.ndarray A: the matrix to be factored
    :param int LDA: the leading dimension of A
    :param int N: the order of the matrix  A
    :param numpy.ndarray IPVT: the (integer) pivot vector from dgeco or dgefa
                               shape (N,)

    :param int JOB: 1 both determinant and inverse
                    2 inverse only
                    3 determinant only

    .. note:: 1) arrays are modified in place.
              2) on return, A  will be inverse of original matrix if requested
                            otherwise unchanged.
              3) DET and WORK are not used outside
                :param float DET: determinant of original matrix if requested.
                                  otherwise not referenced.
                                  determinant = det(1) * 10.0**det(2)
                                  with 1.0 <= abs(det(1)) < 10.0, or det(1) == 0.0
                :param numpy.ndarray WORK: a vector of which the contents will be destroyed.
                               shape (N,)

              4) Error condition: a "division by zero" will occur if the input factor
                                  contains a zero on the diagonal and the inverse is requested.
                                  It will not occur if the functions are called correctly
                                  and if dgeco has set (achieve?) RCOND > 0.0,
                                  or dgefa has set (achieve?) INFO == 0
    """
    DET = np.zeros(2)
    WORK = np.zeros(N)

    # Compute determinant
    if JOB == 1 or JOB == 3:
        DET[0] = 1.0
        DET[1] = 0.0
        TEN = 10.0
        for i in range(1, N + 1):
            if IPVT[i] != i:
                DET[0] = -DET[0]
            DET[0] = A[i, i] * DET[0]

            if DET[0] == 0.0:
                break

            while abs(DET(1)) < 1.0:
                DET[0] *= TEN
                DET[1] -= 1.0

            while abs(DET(1)) >= TEN:
                DET[0] /= TEN
                DET[1] += 1.0

    # Compute inverse(U)
    if JOB == 3:
        return

    for K in range(N):
        A[K, K] = 1.0 / A[K, K]
        T = -A[K, K]
        A[: K - 1, K] *= T  # DSCAL[K-1,T,A(1,K),1)
        KP1 = K + 1
        if N >= KP1:
            for J in range(KP1 - 1, N):  # in original Fortran: from KP1 to N (included)
                T = A[K, J]
                A[K, J] = 0.0
                A[:K, J] += T * A[:K, K]  # DAXPY[K,T,A(1,K),1,A(1,J),1)

    # form inverse(U)*inverse(L)
    NM1 = N - 1
    if NM1 < 1:
        return

    for KB in range(NM1):
        K = N - 1 - KB
        KP1 = K + 1
        for i in range(KP1, N):
            WORK[i] = A[i, K]
            A[i, K] = 0.0
        for J in range(KP1, N):
            T = WORK(J)
            A[:N, K] += T * A[:N, J]  # DAXPY(N,T,A(1,J),1,A(1,K),1)
        L = IPVT[K]
        if L != K:
            A[:, K], A[:, L] = A[:, L], A[:, K].copy()  # DSWAP(N,A(1,K),1,A(1,L),1)

    return
