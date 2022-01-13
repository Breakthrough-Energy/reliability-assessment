
def thetac(THET,THETC,LT,NN):
    """
    Generates vector of bus angles for all buses including ref bus
    
    :param numpy.ndarray THET: angle at node i of the ADM matrix
    :param numpy.ndarray THETC: angle for actual node corresponding to node i of the ADM matrix
    :param numpy.ndarray LT: actual node corresponding to node i of the ADM matrix
    :param int NN: number of nodes
    
    :return: (*None*)
    """
    NX=NN-1
    
    for i in range(NN):
        THETC[i] = 0.
    
    for i in range(NX):
        j = LT[i]
        THETC[j] = THET[i]
    
    return # THETC; no need for explictly return since it is array (ref type)