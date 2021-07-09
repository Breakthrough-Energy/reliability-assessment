# -*- coding: utf-8 -*-

import globaVar as g 

def xldnew(FileNameAndPath): 
    """
     Read and parse raw text data from the input file LEEI; and then
     normalize and re-scale the parsed hourly load data of each area

    :param string FileNameAndPath -- full file path for the input file 'LEEI '
    :param array HRLOAD: 2D array of hourly load data 
                            (0-dim: area index, 1-dim: hour index (1~8760hr))
    :return: (*arrays*) 
    
     Note: the hardcoded number "8760" is OK in this code package. 
     Because for Monte Carlo Simulation in reliability analysis area, researchers does not 
     try to simulate "real-life year", but purely for statistic purpose. In fact, 
     each reliability index is eventually an averaged number. So, the leap year is not
     a concern in this area.
    """
   
    # g.HRLOAD = zeros((20,8760)) has initially been executed in globalVar.py
    # but re-initialization may be needed here (will check later)
    # please keep this comment here.
    
    # Read file LEEI and parsing
    FileNameAndPath = 'LEEI'
    file =  open(FileNameAndPath, 'r')
    Lines = file.readlines()    

    idx = 0
    while (idx < len(Lines)):
        while idx < len(Lines)-1 and Lines[idx][:4] != 'AREA':
            idx = idx + 1 
        areaIdx = int(Lines[idx][6]) - 1 # index from '0' in Python!
        hourIdx = 0
        while (hourIdx < 8760):
            idx = idx + 1
            tmp = list(map(float, Lines[idx].strip().split()))
            g.HRLOAD[areaIdx, hourIdx:hourIdx+12] = tmp[-12:]
            hourIdx = hourIdx + 12
        idx = idx + 1 
    
    # Normalize and re-scale the hourly load data of each area
    for i in range(g.NOAREA): # e.g., g.NOAREA = 5
        XMAX = max(g.HRLOAD[i,:])
        XMAX = max(XMAX, 0.001)
        # normalizing
        g.HRLOAD[i,:] = g.HRLOAD[i,:] / XMAX
        # re-scaling
        g.HRLOAD[i,:] = g.HRLOAD[i,:] * g.PKLOAD[i]

        
    
                      
    