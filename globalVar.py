# -*- coding: utf-8 -*-

from numpy import ones, zeros

'''
Global Variables and/or their initial values

Note: 

    1) This file is the global variable definitions file 
    for a under-developing reliability assessment code package. 
    To avoid messy and repeated annotations, the meaning of each variable/array 
    will be explained in the functions where they are used.
    
    2) This file will get continuously updating based on developers' 
    progress and understanding.
'''

PLNDST = ones((600,1))*1.0
CADJ = ones((600,1))*0.0
SADJ = ones((20,1))*0.0
LSFLG = ones((20,1))*0.0
BLPA = ones((1800,1))*0.0
SUSTAT = ones((96,1))*0.0

LOLTHA = ones((20,5))*0.0
LOLGHA = ones((20,5))*0.0
MGNTHA = ones((20,5))*0.0
MGNGHA = ones((20,5))*0.0
LOLTPA = ones((20,5))*0.0    
LOLGPA = ones((20,5))*0.0    
MGNTPA = ones((20,5))*0.0  
MGNGPA = ones((20,5))*0.0       
   
LOLTHP = ones((5,1))*0.0
LOLGHP = ones((5,1))*0.0
MGNTHP = ones((5,1))*0.0
MGNGHP = ones((5,1))*0.0
LOLTPP = ones((5,1))*0.0    
LOLGPP = ones((5,1))*0.0    
MGNTPP = ones((5,1))*0.0  
MGNGPP = ones((5,1))*0.0   

SOLTHA = ones((20,5))*0.0
SOLGHA = ones((20,5))*0.0
SGNTHA = ones((20,5))*0.0
SGNGHA = ones((20,5))*0.0
SOLTPA = ones((20,5))*0.0    
SOLGPA = ones((20,5))*0.0    
SGNTPA = ones((20,5))*0.0  
SGNGPA = ones((20,5))*0.0  
SOLSHA = ones((20,5))*0.0    
SGNSHA = ones((20,5))*0.0    
SOLSPA = ones((20,5))*0.0  
SGNSPA = ones((20,5))*0.0      

SOLTHP = ones((5,1))*0.0
SOLGHP = ones((5,1))*0.0
SGNTHP = ones((5,1))*0.0
SGNGHP = ones((5,1))*0.0
SOLTPP = ones((5,1))*0.0    
SOLGPP = ones((5,1))*0.0    
SGNTPP = ones((5,1))*0.0  
SGNGPP = ones((5,1))*0.0 
SOLSHP = ones((5,1))*0.0    
SGNSHP = ones((5,1))*0.0    
SOLSPP = ones((5,1))*0.0  
SGNSPP = ones((5,1))*0.0       

HLOLE = ones((20,22))*0.0 
DPLOLE = ones((20,22))*0.0 
EUES = ones((20,22))*0.0 

XNEWA = ones((20,3))*0.0 
XNEWP = ones((3,1))*0.0 
SSQA =  ones((20,3))*0.0 
SSQP = ones((3,1))*0.0 

WOLSHA = ones((20,1))*0.0   
      
NUNITS = 0
RANDG = zeros((600,1))
PROBG = zeros((600,2))
PCTAVL = zeros((600,1))
PLNDST = zeros((600,1))

MFA = 0
EVNTS = zeros((103,1))   
IPOINT = 0

HRLOAD = zeros((20,8760))
PKLOAD = zeros((20,1))