# -*- coding: utf-8 -*-

import numpy as np 
import globalVar as g

def lstate():
    """
    LSTATE establishes the state for each line.
    
    :param: NLINES int -- total number of lines
    :param: RANDL array -- array of random numbers for each line
    :param: PROBL array -- array of predefined capacity probability of each line
    :return: LNSTAT (array) -- array of line state value (in integer) of each line                        
    """    
    
    print("Enter subroutine lstate, total number of lines is %d" % (g.NLINES))
    
    for i in range(g.NLINES):
        # Draw a random number for each line
        g.RANDL[i] = np.random.rand()
        
        # Determine the State and Available Capacity for Each Generator.
        if g.RANDL[i] <= g.PROBL[0,i]:
            g.LNSTAT[i] = 1
        elif g.RANDL[i] <= g.PROBL[1,i]:
            g.LNSTAT[i] = 2
        elif g.RANDL[i] <= g.PROBL[2,i]:
            g.LNSTAT[i] = 3
        elif g.RANDL[i] <= g.PROBL[3,i]:
            g.LNSTAT[i] = 4
        elif g.RANDL[i] <= g.PROBL[4,i]:
            g.LNSTAT[i] = 5
        else:
            g.LNSTAT[i] = 6
            

