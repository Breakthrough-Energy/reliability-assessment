import numpy as np 

def gstate(NUNITS, PROBG, RATING, DERATE, PLNDST):
    """
    It samples the state of available capacity for each generator up to NUNITS
    
    :param int NUNITS: total number of generator units
    :param array PROBG: 2D array of accumulated probability for each capacity tier of each unit
    :param array DERATE: array of the derated capacity of each unit
    :param array RATING:  array of the fully rated capacity of each unit
    :param array PLNDST: array of the on/off status (due to planned maintenance) of each unit
    
    :return: PCTAVL (array) -- array of unadjusted power capacity (in p.u. value) of each unit
             AVAIL (array) -- array of the finalized power capacity (in nominal value) 
                              of each unit             
    Note: neither rng.choice nor vectorization trick is appropriate here due to 
          the following reasons:
          1) The capacity probability of each gen unit CAN BE different with each other’s. 
             That is why the PROBG array is originally designed as 2-D but not 1-D.
          2) The rng.choice function cannot (or can but awkwardly) adapt to the situation here due to 1); 
          3) Most of all: generating a random vector and applying vectorized operation may 
             break a critical implicit requirement for the Monte Carlo simulation -- 
             "generated random numbers should be mutually independent."
    """    

    print("Enter subroutine gstate, total number of Gen units is %d" % (NUNITS))
    
    PCTAVL = np.zeros((NUNITS,1))
    AVAIL = np.zeros((NUNITS,1))
                        
    for i in range(NUNITS):
        # Draw a random number for each unit
        rand_number = np.random.rand()
        
        # Determine the State and Available Capacity for Each Generator.
        if rand_number <= PROBG[i,0]:
            PCTAVL[i] = 1.0       # normalized value for gen. capacity is used here
        elif rand_number <= PROBG[i,1]:
            PCTAVL[i] = DERATE[i]   # normalized value for gen. capacity is used here
        else:
            PCTAVL[i] = 0
            
        AVAIL[i] = PCTAVL[i] * RATING[i] * PLNDST[i]
            
    return PCTAVL, AVAIL, RANDG
