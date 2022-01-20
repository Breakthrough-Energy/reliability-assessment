
import numpy as np

def hour(MFA, ATRIB, NUMINQ, IPOINT, EVNTS):
    
    # ? LOAD = (20,1) never used in this subroutine
    IFLAG, JFLAG  = 0,0
    
    ATRIB(1) = CLOCK + JSTEP

    # may need modify here: some retun var can be implciit
    EVNTS, NUMINQ, MFA, IPOINT = filem(MFA, ATRIB, NUMINQ, IPOINT, EVNTS)
    
    print("Entering subroutine hour")
    
    # ..............UPDATE HOUR NUMBER.............
    DXX = 8760.
    JHOUR = CLOCK % DXX # DMOD(CLOCK,DXX)
    if JHOUR == 0:
        JHOUR=8760
    JJ = JHOUR % JFREQ
    
    # need to double check
    JDAY = floor(JHOUR/24) + 1  # (JHOUR-.1)/24 + 1 
    
    if JDAY > 365:
        JDAY = 365
    
    JHOURT=JHOUR
    if JSTEP == 24:
        JHOUR = MXPLHR(JDAY)  # MXPLHR stores the pool peak hour
        
    # ......Draw generator statuses and sum capacities by area...............
    if JJ == 0:
        JJ=1
    
    if JJ == 1. or JFREQ == 1:
        GSTATE
	    SUMCAP
	    LSTATE
	    CALLS = 0.
	    FINDTN

    # STMULT(I,1) IS FORWARD DIRECTION ADJUSTMENT
    # STMULT(I,2) IS BACKWARD DIRECTION ADJUSTMENT
    for i in range(NLINES):
        for j in range(2):
             STMULT[i,j] = 1.0  

	JPNT = 1
    
    for i in range(MXCRIT):
        if JCRIT[JPNT] == 0:
            JPNT = JPNT+1
        else: 
            NOCRIT = JCRIT[JPNT+1]
            CRTVAL = 0.0
            for ii in range(NOCRIT):
                if JCRIT[JPNT+1+II] > 0 and JCRIT[JPNT+1+II] <= 500-1:
                    # maybe 500 - 1 , need to check back later
                    CRTVAL = CRTVAL + AVAIL[JCRIT[JPNT+1+II]]
                else:
	                if IGENE == 0:
                        print("Logic error...")
                    IGENE=1
                    
            if CRTVAL == 0.0:
                STMULT(JCRIT(JPNT),1)=FLOAT(JCRIT(JPNT+NOCRIT+3))/100.
                STMULT(JCRIT(JPNT),2)=FLOAT(JCRIT(JPNT+NOCRIT+4))/100. 
            
            JPNT = JPNT+NOCRIT+5  
            
    
    for NST in range(NFCST):
        
        CALL FINDLD(NST)
        
        FLAG = 0.0
	    FLAG2 = 0.0
        
        for j in range(NOAREA):
            MARGIN[j] = int(SYSCON[j] - CAPREQ[j]) # doublec ehck  IFIX usage !
            #  IFIX(SYSCON[j] - CAPREQ[j]) 
            JMJ = int(SYSOWN[j] - CAPREQ[j]) # IFIX(SYSOWN[j] - CAPREQ[j])
            if JMJ < 0:
                FLAG2 = 1.
            if MARGIN[J] < 0:
                FLAG = FLAG + 1.0
                IT = J
                
        if FLAG == 0.0:
            return
        
        NSHRT = int(FLAG) # IFIX(FLAG)
        if NSHRT == NOAREA:
            CALL HRSTAT(NST)
            if JHOUR == MXPLHR[JDAY]:
                CALL PKSTAT(NST)
        else:
            if FLAG > 1.0:
                continue
            else:
                NEED = abs(MARGIN[IT])
                for j in range(NOAREA):
                    if MARGIN[j] > 0:
                        N = LINENO[J,IT]
                        if N > 0:
                            M = (LNSTAT[N]-1)*3+2
                            if LP[N,2] == j:
                                XX = BLPA[N,M]
                                LNCAP[N] = int(XX) # IFIX(XX)
                            else:
                                XX = BLPA[N,M+1]
                                LNCAP[N] = int(XX) # IFIX(XX)
                            MXHELP = MIN(MARGIN(J),LNCAP(N))
                            NEED = NEED - MXHELP
            
            for j in range(NOAREA):
                MARGIN[j] < 0:
                    TEMPFL = 1. 
            
            JFLAG=0
            CALL TRANSM(IFLAG,JFLAG)
            CALLS = 1.
            if IFLAG ==. 0:
                IFLAG = 1
            if JFLAG == 0:
                return
            TEMPFL=0.  
            
            for j in range(NOAREA):       
    	       ZZ =  -SADJ[j]
    	       MARGIN[j] = int(ZZ) # IFIX(ZZ)  
               if MARGIN[j] < 0:
                   TEMPFL = 1.
     	    
            if TEMPFL == 0:
                return
    
    return 
    # Note: a dummy 'return' is delibrately added here to remind myself that the code logic here is complete, not half done. We can alwasy remore it before merge or before create a formal PR.
        

        
        