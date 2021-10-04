import numpy as np

def admitb(LP, BLP):
    """
    :param np.ndarray LP: line connection data
        LP[L,K]: L, ENTRY NUMBER; K=1 line number,
                  K=2 starting area, K=3 ending area.
    :param np.ndarray BLP: (adjusted) line data:
        BLP[I,J]: Six sets of three entries - admitance,forward and 
              backward capacities for six states.
              (refer to file 'INPUTB' for more details)

    :return: (*np.ndarray*) BB -- 2D array of the 'B matrix' (i.e., admittance matrix) 
                                  size: # Area-by-# Area
    """
    #BLP = zeros((100,3))
    #LP = zeros((100,3))
    BB = np.zeros((20,20)) # An exilicit 'NAREA' parameter maybe used here
    
    for i in range(()):
        for j in range(()):
            BB[i,j] = 0.0
    
    i = 0
    while(LP[i,0] != 0): # End of data detected by line #0
        j = LP(i,1) # from-bus id
        k = LP(i,2) # to-bus id
        BB[j,k] = BB[j,k] + BLP[i,0] * (-1.0)
        BB[k,j] = BB[j,k]
        BB[j,j] = BB[j,j] + BLP[i,0]
        BB[k,k] = BB[k,k] + BLP[i,0]
        i = i + 1

    return BB
    
    
    
    
    