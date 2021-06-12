# -*- coding: utf-8 -*-

import numpy as np

def readInputFileB(FileNameAndPath):
    """ read and parse raw text data from the input file INPUTB.

    :param string FileNameAndPath: absolute file path and name for the file 'INPUTB'

    :return: (*arrays*) -- arrays for each data card (raw data); 
             each data card can have multiple arrays
    """
    file =  open(FileNameAndPath, 'r')
    Lines = file.readlines()
   
    card_flag = 0
    cnt = 0;
    for line in Lines:
        cnt = cnt + 1;
        if line[:2] != 'ZZ':
            continue
        else:
            card_flag = 1 
            break
       
    if not card_flag:
        print('No data card found in INPUTB file!')
        exit
    
    '''----------------------------------------------------------
    ZZTC: title card
    The purpose of this card is to vrovide title of the ontnont.
    Data inserted under this card is simply reproduced. This is the
    only formatted card in this file 
    '''
    if line[:4] == 'ZZTC':
        cnt = cnt + 1
        line = Lines[cnt]  
        while line != '\n': 
            line_no = int(line[:3].strip())
            assert line_no <= 20 # if total lines are more than 20, aborted
            print(line); # e.g. printf(' 1 THIS IS A FIVE AREA SYSTEM ')
            cnt = cnt + 1
            line = Lines[cnt]   
    else:
        print('Title card (ZZTC) not exist in INPUTB file!')
        exit()
        
     
    '''------------------------------------------------------------
    ZZMC: Miscellaneous card. 
    '''         
    while cnt < len(Lines) and Lines[cnt][:2] != 'ZZ':
        cnt = cnt + 1
    if cnt == len(Lines) or Lines[cnt][:4] != 'ZZMC':
        print('Miscellaneous card (ZZMC) not exist in INPUTB file!')
        exit()
        
    cnt = cnt + 4
    tmp = list(map(float, Lines[cnt].strip().split()))
    '''
    The basic seed used by the program to create seeds for geenrators
    and transmiison lines. a 7-digit int
    
    Note: may be useless in python sicne random number can be 
    easily generated. Will remove it later
    '''
    JSEED = int(tmp[0]) # e.g. 345237
    
    ''' 
    Load-loss sharing: 
    0 --"Yes"; 1 --"No"
    '''
    NLS = int(tmp[1]) #1
    
    '''
    Ending weeks for the 1st (begining week is assumed to be thtis 1st week),
    2nd, 3rd seasons. The ending weeks of the 4th season is assumed as 52
    '''
    # e.g.  #[13, 26, 39]
    IW1, IW2, IW3= int(tmp[2]), int(tmp[3]), int(tmp[4])
    # ----------Vars for convergence test-------------- 
    '''
    1: based on area-statistics. The area is specified by the variable KVL below
    2: based on pool statistics
    '''
    KWHERE = int(tmp[5]) #e.g. 1
    
    ''' 
    1: Hourly statistics are to be used for convergence 
    2: Peak statistic are to be used for convergence'''
    KVWHEN = int(tmp[6]) #e.g. 1
    
    ''' 
    1: LOLE is to be used for convegence 
                hourly LOLE for 'WHEN = 1'
                peak LOLE for 'WHEN = 2'
    2: EUE is to be used for convegence ( for 'WHEN = 1' only)        
    '''
    KVSTAT = int(tmp[7])#e.g. 1
    
    '''
    1: Weighted average statistcs (cosniderring forecast error) 
       is to be used for convegence 
    2: Use only the No-forecast-error statistics  
    '''
    KVTYPE = int(tmp[8])#e.g. 2
    
    '''
    Area number (used only if 'WHERE'=1) for convergence
    '''
    KVLOC = int(tmp[9])#e.g. 1
    
    '''
    0: Program stops when standard deviation of the 
    tested statisticcs < (0.025 * mean); i.e. the true valye lies 
    betyween pos/neg 5% of the estiamte with confidence level at 95%.
    
    >0:  Program stops when standard deviation of the 
    tested statisticcs < (CVT * mean); i.e. the true valye lies 
    betyween pos/neg (2 CVT)% of the estiamte with confidence level at 95%.
    '''
    CVTEST = float(tmp[10]) #e.g. 0.025
    
    '''
    Time to terminate simualtion if no convergence. 
    SHould be an INT type (years)
    '''
    FINISH = int(tmp[11])#e.g. 9999
    
    # ---- Other variables 
    '''
    1:  Stats are collected every hour
    24: Daily peak stats only are collected 
    '''
    JSTEP = int(tmp[12])#e.g. 1
    
    '''
    1:  Hourly Monte Carlo draws of generator ;and line status are used
    24: Daily Monte Carlo draws are used 
    '''
    JFREQ = int(tmp[13])#e.g. 1
    
    '''
    Defines the upper limit of the prob distribution of 
    Expected Unserved Energy, in MWHRS.
    Note: the roginal Fortran code set it as int type, 
    but float type is used here since it may be more reasonable
    '''
    MAXEUE = float(tmp[14]) #e.g. 1000
    
    '''
    0: Transmission mod results are not printed.
    1: Transmission mod results are printed into file TRAOUT.
    '''
    IOI = int(tmp[15])#e.g. 0
    
    '''
    0: Only final statistics are printed.
    1: In addition to final statistics, results after each 
       replication are also printed.
    '''
    IOJ = int(tmp[16])#e.g. 0
    
    '''
    0: Delete DUMP after the run.
    1: Retain DUMP after the run.
    '''
    IREM = int(tmp[17])#e.g. 1
    
    '''
    Interval, in number of replications, for storing
    '''
    INTV = int(tmp[18])#e.g. 5
   
    '''
    0: Input data from INPUTB and INPUTC is not printed.
    1: Input data is printed.'''
    IREPD = int(tmp[19])#1
    
    '''
    0: Planned outage schedules are not printed.
    1: Planned outage schedules are printed.
    '''
    IREPM = int(tmp[20])#e.g. 1



    '''------------------------------------------
    ZZLD:System data card. Area name in four letters.
    The purpsoe of this card is to speicify data which is 
    applicable to a given area.                     
    '''              
    while cnt < len(Lines) and Lines[cnt][:2] != 'ZZ':
        cnt = cnt + 1
    if cnt == len(Lines) or Lines[cnt][:4] != 'ZZLD':
        print('System data card (ZZLD) not exist in INPUTB file!')
        exit()
        
    cnt = cnt + 4
    II = 0
    # assume maximum number of areas = 1000
    SNRI, NAR = np.zeros((1000,1)).astype(int), ["" for _ in range(1000)]
    RATES, ID = np.zeros((1000,3)), np.zeros((1000,4)).astype(int)
    
    while cnt < len(Lines) and len(Lines[cnt].strip()) > 0 and Lines[cnt].strip()[0].isdigit():
        tmp = Lines[cnt].strip().split()
        cnt = cnt + 2
        
        '''
        Serial Number.
        The serial number in this and subsequent types of cards identifies 
        each entry uniquely and specifies the sequence of data entries.
        After reading, the data entries are arranged by the program in the 
        sequence specified under SN. The serial number can have up to three
        decimal places. This he.lps in deleting lines and inserting new lines
        through INPUTC file.
        '''
        SNRI[II] = int(tmp[0].strip('.'))
        
        '''
        The name of the area. It can have up to four letters and 
        must be enclosed by quotes.
        '''
        NAR[II] = tmp[1]
        
        '''
        Annual peak in the area, in MW.
        '''
        RATES[II, 0] = float(tmp[2])
        
        '''
        Load forecast uncertainty (LFU).
        One standard deviation expressed as percentage of the mean
        '''
        RATES[II, 1] = float(tmp[3])
        
        '''
        Outage Window.
        The weeks during which planned maintenance can be performed.
            BEG WK: Beginning week of the outage window
            END WK: Ending week of the outage window
        '''
        ID[II, 0], ID[II, 1] = int(tmp[4]), int(tmp[5])

        '''
        Forbidden Period.
        A part of the outage window during which planned maintenance 
        is NOT allowed.
            BEG WK: Beginning week of the Forbidden Period.
            END WK: Ending week of the Forbidden Period.
        '''
        ID[II, 2], ID[II, 3] = int(tmp[6]), int(tmp[7])
        
        '''
        Sum of Flows Constraint.
        The algebraic sum of flows at this node is not to exceed this value. 
        '''
        RATES[II, 2] = float(tmp[8])
        
        II = II + 1 
    
    SNRI =  SNRI[:II].flatten() # or usign 'squeeze()'
    NAR = NAR[:II]
    RATES = RATES[:II]
    ID = ID[:II]
    
     
    '''------------------------------------------------------------
    ZZUD: GEN UNIT DATA 
    '''      
    while cnt < len(Lines) and Lines[cnt][:2] != 'ZZ':
        cnt = cnt + 1
    if cnt == len(Lines) or Lines[cnt][:4] != 'ZZUD':
        print('Gen.unit data card (ZZUD) not exist in INPUTB file!')
        exit()
        
    cnt = cnt + 13
    II = 0
    
    # assume maximum number of Gen Units = 1000
    SNRI_ZZUD = np.zeros((1000,1)).astype(int)
    NAT, NAR_ZZUD = ["" for _ in range(1000)], ["" for _ in range(1000)]
    HRLOAD, ID_ZZUD = np.zeros((7, 1000)), np.zeros((1000, 5)).astype(int)
    # Note: the orginal Fortran code set the HRLOAD array as INT type; 
    # Here, I use float type for it.
    
    while cnt < len(Lines) and len(Lines[cnt].strip()) > 0 and Lines[cnt].strip()[0].isdigit():

        tmp = Lines[cnt].strip().split()
        cnt = cnt + 1
        
        '''
        Serial number (refet to ZZLD)
        '''
        SNRI_ZZUD[II] =  int(tmp[0].strip('.'))
        
        '''
        Unit name is six alphanumeric characters. The 
        first four letters are the plant number and the 
        next two numbers identify the unit.
        '''
        NAT[II] = tmp[1]
        
        '''
        Area of location, up to four letters.
        '''
        NAR_ZZUD[II] =  tmp[2]
        
        '''
        Unit capacity, MW, in the ith season (see ZZLD)        
        '''
        # CAP1, CAP2, CAP3, CAP4 
        HRLOAD[0:4, II] = list(map(float, tmp[3:7]))

        '''
        Derated forced outage rate (DFOR)
        '''
        HRLOAD[4, II] = float(tmp[7])       
        
        '''
        Forced outage rate(FOR)
        '''
        HRLOAD[5, II] = float(tmp[8])       
             
        '''
        Percent derating (DER) due to partial failure. This is
        different from the seasonal derating which can be
        specified using CAPI.        
        '''         
        HRLOAD[6, II] = float(tmp[9])       
          
        '''
        Predetermined (1) or automatic (0) scheduling.
        '''
        ID_ZZUD[II, 0] = int(tmp[10])       
        
        # Notes:
        #    1. In scheduling the planned outages, the program does not
        #       take out two units belonging to one plant at the same
        #       time. Also the program can accept up to two planned
        #       outages per year for each unit.
        
        #    2. The program schedules units for planned outage by
        #       adjusting them into valleys in the load cycle. Therefore
        #       for any area having only generation, planned outages must
        #       be predetermined by the user.
        #
        #    3. If the program cannot schedule a unit in the automatic
        #       mode, it will print a message to indicate this and
        #       schedule the unit using B1,D1l and B2,D2 parameters.
        '''
        Beginning week and duration of the first outage.
        In the automatic mode Bl is ignored.(B1, D1)
        '''      
        ID_ZZUD[II, 1:3] = list(map(int, tmp[11:13])) 

        '''
        Beginning week and duration of second outage. In 
        the automatic mode, B2 is ignored.(B2, D2)
        '''
        ID_ZZUD[II, 3:5] = list(map(int, tmp[13:15]))          
        
        II = II + 1 

    SNRI_ZZUD =  SNRI_ZZUD[:II].flatten() 
    NAT = NAT[:II]
    NAR_ZZUD = NAR_ZZUD[:II]
    HRLOAD = HRLOAD[:, :II]
    ID_ZZUD = ID_ZZUD[:II, :]
    
    
    '''------------------------------------------------------------
    ZZFC: Firm Contracts data
    This card is for specifying the firm interchanges of power,
    between areas. The following format is used for input of
    these interchanges:  
    '''         
    while cnt < len(Lines) and Lines[cnt][:2] != 'ZZ':
        cnt = cnt + 1
    if cnt == len(Lines) or Lines[cnt][:4] != 'ZZFC':
        print('Unit Firm Contracts data card (ZZFC) does not exist in INPUTB file!')
        exit()
    '''
    No concrete ZZFC data is given. Leave as a placeholder here at this moment.
    '''
    cnt = cnt + 2
    
     
    
    '''------------------------------------------------------------
    ZZOD: Unit Ownership data
    This card is for specifying the joint ownership of a unit by several areas. 
    '''         
    while cnt < len(Lines) and Lines[cnt][:2] != 'ZZ':
        cnt = cnt + 1
    if cnt == len(Lines) or Lines[cnt][:4] != 'ZZOD':
        print('Unit ownership data card (ZZOD) does not exist in INPUTB file!')
        exit()
    '''
    No concrete ZZOD data is given. Leave as a placeholder here at this moment.
    '''        
    cnt = cnt + 4
    


    '''------------------------------------------------------------
    ZZTD: Line Data
    The purpose of this card is to specify the data for
    transmission links.     
    '''         
    while cnt < len(Lines) and Lines[cnt][:2] != 'ZZ':
        cnt = cnt + 1
    if cnt == len(Lines) or Lines[cnt][:4] != 'ZZTD':
        print('Line data card (ZZTD) does not exist in INPUTB file!')
        exit()

    cnt = cnt + 8
    
    # assume maximum number of Lines = 1000
    SNRI_ZZTD = np.zeros((1000,1)).astype(int)
    ID_ZZTD  =  np.zeros((1000,1)).astype(int)
    NAR_ZZTD, NAE = ["" for _ in range(1000)], ["" for _ in range(1000)]
    ADM = np.zeros((1000, 6))
    CAP, CAPR = np.zeros((1000, 6)), np.zeros((1000, 6))
    PROP = np.zeros((1000, 6))
    
    II = 0
    while cnt < len(Lines) and len(Lines[cnt].strip()) > 0 and Lines[cnt].strip()[0].isdigit():
        tmp = Lines[cnt].strip().split()
        
        SNRI_ZZTD[II] = int(tmp[0].strip('.'))
        ID_ZZTD[II] =  int(tmp[1].strip('.'))
        
        '''
        Name of From area
        '''
        NAR_ZZTD[II] = tmp[2]
        
        '''
        Name of To area
        '''    
        NAE[II] = tmp[3]
        
        ADM[II,0] = float(tmp[4])
        CAP[II,0] = float(tmp[5])
        CAPR[II,0] = float(tmp[6])
        PROP[II,0] = float(tmp[7])
        for i in range(1, 6):
            cnt = cnt + 1
            tmp_ = Lines[cnt].strip().split()
            ADM[II,i] = float(tmp_[0])
            CAP[II,i] = float(tmp_[1])
            CAPR[II,i] = float(tmp_[2])
            PROP[II,i] = float(tmp_[3])
        
        cnt = cnt + 1
        II = II + 1 
    
    SNRI_ZZTD =  SNRI_ZZTD[:II].flatten() 
    ID_ZZTD =  ID_ZZTD[:II].flatten() 
    NAR_ZZTD = NAR_ZZTD[:II]
    NAE = NAE[:II]
    ADM = ADM[:II, :]
    CAP, CAPR = CAP[:II, :], CAPR[:II, :]
    PROP = PROP[:II, :]              
    
    
    '''------------------------------------------------------------
    ZZDD: Line derarting data
    Status of certain units can effect the line rating. For a
    given line, four units can be specified in a combination such that
    if all the units in the combination are down, then the line ratings
    will be multiplied by the derating factors. For a given line, more
    than one combination can be specified.    
    '''   
    '''
    No concrete ZZOD data is given. Leave as a placeholder here at this moment.
    '''     
      
    '''------------------------------------------------------------
    ZZND:TERMINATING CARD (The end of INPUT-B file)      
    ''' 
    return JSEED,NLS,IW1,IW2,IW3,KWHERE,KVWHEN,KVSTAT,KVTYPE,KVLOC,\
        CVTEST,FINISH,JSTEP,JFREQ,MAXEUE,IOI,IOJ,IREM,INTV,IREPD,IREPM,\
        SNRI,NAR,RATES,ID,\
        SNRI_ZZUD, NAT, NAR_ZZUD, HRLOAD, ID_ZZUD, \
        SNRI_ZZTD, ID_ZZTD, NAR_ZZTD, NAE, ADM, CAP, CAPR, PROP
