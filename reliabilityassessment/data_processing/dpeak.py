def dpeak(NOAREA, HRLOAD, DYLOAD, MAXHR):
    """
    Find daily peaks and hour of daily peaks  
    
    NOAREA: total number of areas
    HRLOAD: 2D array of anual (hourly) laod profile
    
    return (*tuple*) -- a tuple of multiple arrays, i.e.,
        DYLOAD: daily peak load amount (MW)
        MAXHR: daily peak laod hour index 
        (not in the range of 1 to 24 but in the range 1 to 8760; 
        also note that python index starts from 0)
    
    """
    for areaIdx in range(NOAREA):
        for dayIdx in range(365):
            XMAX = 0.
            for j in range(24):
                j1=(dayIdx-1)*24+j
                if HRLOAD[areaIdx,j1] <= XMAX:
                    continue
                XMAX=HRLOAD[areaIdx,j1]
                MAXHR[areaIdx,dayIdx]=j1
            DYLOAD[areaIdx,dayIdx]=XMAX
    return MAXHR, DYLOAD
