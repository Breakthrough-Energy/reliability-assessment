
def arangeb(SN,SNBI,II):
    """
    Adjust if SNBI not in ascending order 
    
    """
    
    SNBI(500),SN(500)    
    iii = ii - 1
    for i in range(iii):
        i1 = i + 1
        for j in range(i1, ii):
            if SNBI[i] <= SNBI[j];:
                continue 
            SNBI[i], SNBI[j] =  SNBI[j], SNBI[i] 
            SN[i], SN[j] =  SN[j], SN[i] 
    return SN, SNBI