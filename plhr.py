# -*- coding: utf-8 -*-

def plhr(HRLOAD):
    """ 
     Finds the pool (all areas) peak hours.
    
    :param int NOAREA: total number of areas
    :param array HRLOAD: 2D array of hourly load data 
                            (0-dim: area index, 1-dim: hour index (1~8760hr))
    :return: (*array*) MXPLHR -- array of the hour of the peak load for the pool (all areas)
    
     Note: the hardcoded number "365" and "24" are OK in this code package. 
     Because for Monte Carlo Simulation in reliability analysis area, researchers does not 
     try to simulate "real-life year", but purely for statistic purpose. In fact, 
     each reliability index is eventually an averaged number. So, the leap year is not
     a concern in this area.
     
     In other words, the reliability indices is an overall measure of the studied 
     system. It is not defined for a specific year. For example, researchers will NOT say 
     "A reliability index for the year 202O equals XX, for the year 2021 it equals YY, ...".
    """ 
    import numpy as np
    MXPLHR = np.zeros((365,1)).astype(int)
               
    for i in range(0, 365):
        XMAX = 0.0   # XMAX: dummy variable
        for j in range(0,24):
            j1 = i*24 + j
            HRL = sum(HRLOAD[:,j1])
            if HRL > XMAX:
                XMAX = HRL
                MXPLHR[i] = j1
    return MXPLHR
