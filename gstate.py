# -*- coding: utf-8 -*-

import numpy as np 
import globalVar as g

def gstate():
    """
    GSTATE determines the available capacity for each generator up to NUNITS
    
    :param: NUNITS int -- total number of generator units
    :param: RANDG array -- array of random numbers for each unit
    :param: PCTAVL array -- array of unadjusted power capacity (in p.u. value) of each unit
    :param: PROBG array -- array of predefined capacity probability of each unit
    :param: DERATE array -- array of the derated capacity of each unit
    :param: RATING array -- array of the fully rated capacity of each unit
    :param: PLNDST array -- array of the on/off status (due to planned maintenance) 
                            of each unit
    :return: AVAIL (array) -- array of the finalized power capacity (in nominal value) 
                              of each unit
    """    

    print("Enter subroutine gstate, total number of Gen units is %d" % (g.NUNITS))
    
    for i in range(g.NUNITS):
        # Draw a random number for each unit
        g.RANDG[i] = np.random.rand()
        
        # Determine the State and Available Capacity for Each Generator.
        if g.RANDG[i] <= g.PROBG[i,0]:
            g.PCTAVL[i] = 1.0       # Note: use per uinit value for gen. capacity
        elif g.RANDG[i] <= g.PROBG[i,1]:
            g.PCTAVL[i] = g.DERATE[i] # Note: use per uinit value for gen. capacity
        else:
            g.PCTAVL[i] = 0
            
        g.AVAIL[i] = g.PCTAVL[i] * g.RATING[i] * g.PLNDST[i]
            
