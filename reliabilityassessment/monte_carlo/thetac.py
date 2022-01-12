
def thetac(THET,THETC,LT,NN):
    """
    This subroutine generates vector THETC of bus angles.
    for all buses including ref bus.
    
    THET(I) CONTAINS ANGLE AT NODE I OF THE ADM MATRIX
    LT(I) CONTAINS THE ACTUAL NODE CORRESPONDING TO NODE
    I OF THE ADM MATRIX
    
    return THETC
    """
    NX=NN-1
    
    for i in range(NN):
        THETC[i] = 0.
    
    for i in range(NX):
        j = LT[i]
        THETC[j] = THET[i]
    
    return # THETC; no need for explictly return since it is array (ref type)