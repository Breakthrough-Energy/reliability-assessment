def admitb(LP, BLP, BB):
    """
    Construct the admittance matrix (B) from the line data array (BLP)

    :param numpy.ndarray LP: array storing line related information
                            shape: (NLINES, 3)
                            LP(I,J)  I  ENTRY  NUMBER
                            LP(I,1)  LINE NUMBER SET=I
                            LP(I,2)  STARTING NODE
                            LP(I,3)  ENDING NODE
    :param numpy.ndarray BLP: contains the data on admittance, cap and backward cap
                              of the line at position I in LP(I,J) (at the state-1)
                              shape: (NLINES, 3)
                              BLP(I,J) I  ENTRY  NUMBER
                              BLP(I,1)  admittance
                              BLP(I,2)  capacity (MW)
                              BLP(I,3)  backward capacity (MW)

    :param numpy.ndarray BB: 2D array of the 'B matrix' (i.e., admittance matrix)
                             used in DC power flow
                             shape (NOAREA, NOAREA)

    :return: (*none*)
    """
    BB[:] = 0.0

    i = 0
    while LP[i, 0] != -1:  # End of data detected by integer -1
        j = LP(i, 1)  # from-bus (area) id
        k = LP(i, 2)  # to-bus (area) id
        BB[j, k] += BLP[i, 0] * (-1.0)
        BB[k, j] = BB[j, k]
        BB[j, j] += BLP[i, 0]
        BB[k, k] += BLP[i, 0]
        i = i + 1

    return BB
