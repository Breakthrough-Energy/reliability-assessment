
import numpy as np 

def findtn(NOAREA, JENT, INTCH, JDAY):
    """
    INTCH is matrix of contract interchanges
    JENT is pointer identifying involved areas
    TRNSFR stores net transfer to (from) rest of pool
    TRANSFERS treated as from higher to lower area number   
    
    JDAY = JHOUR/24+1
    JDAY = (JHOUR-.1)/24 + 1
    IF (JDAY.GT.365) JDAY = 365
    
    return TRNSFR
    """
    
    print('Entering function findtn')  
    
    TRNSFR = np.zeros((20,1))
    
    for i in range(NOAREA):
        TRNSFR[i] = 0.0 
    
    for j2 in range(NOAREA):
        for j1 in range(NOAREA):
            if JENT[j1,j2] > 0:
                JPOINT =  JENT[j1,j2]
                TRNSFR[j1] = TRNSFR[j1] + INTCH[JPOINT,JDAY]
                TRNSFR[j2] = TRNSFR[j2] - INTCH[JPOINT,JDAY]
        
    return TRNSFR
 