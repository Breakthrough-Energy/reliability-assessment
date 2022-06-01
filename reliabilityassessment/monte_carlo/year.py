import numpy as np

from reliabilityassessment.monte_carlo.filem import filem
from reliabilityassessment.monte_carlo.intm import intm 
from reliabilityassessment.monte_carlo.report import report 

def year(IERR,RFLAG, CLOCK, ATRIB, MFA, NUMINQ, IPOINT, EVNTS):
    """

    Parameters
    ----------
    IERR : TYPE
        DESCRIPTION.
    RFLAG : TYPE
        DESCRIPTION.
    CLOCK : TYPE
        DESCRIPTION.
    ATRIB : TYPE
        DESCRIPTION.
    MFA : TYPE
        DESCRIPTION.
    NUMINQ : TYPE
        DESCRIPTION.
    IPOINT : TYPE
        DESCRIPTION.
    EVNTS : TYPE
        DESCRIPTION.

C  EXPLANATION OF VARIABLES USED TO COLLECT AND STORE RELIABILITY STATS
C
C  THESE VARIABLES OCCUR IN THE SECTION OF CODE WHICH FOLLOWS THIS
C
C  ALL VARIABLES HAVE 6-DIGIT IDENTIFIERS.
C
C  THE FIRST THREE DIGITS IDENTIFY THE TYPE OF STATISTIC
C      LOL MEANS THIS IS ANNUAL LOSS OF LOAD STATISTIC OF SOME TYPE
C      MGN MEANS THIS IS ANNUAL SUM OF NEGATIVE MARGINS (EUE)
C      SOL MEANS THIS IS THE CUMULATIVE SUM, ALL YEARS, LOLE
C      SNG MEANS THIS IS THE CUMULATIVE SUM, ALL YEARS, EUE
C
C  THE FOURTH DIGIT IDENTIFIES CAUSE OF OUTAGE
C           T IS TRANSMISSION
C           G IS GENERATION
C           S IS SUM OF T & G
C
C   THE FIFTH DIGIT INDICATES WHETHER THE STAT IS HOURLY OR PEAK
C               H IS FOR HOURLY STATS
C               P IS FOR PEAK STATS
C
C   THE SIXTH DIGIT IS FOR AREA OR POOL
C               A IS AREA
C               P IS POOL
C
C        EXAMPLES
C SGNSPA  IS CUMULATIVE SUM, EUE, TOTAL OF T&G, PEAK, AREA
C LOLTHP  IS ANNUAL LOSS OF LOAD, TRANSMISSION, HOURLY, POOL
C   AND SO ON


    """
    
    XPROB = np.zeros((5))
    
    if NFCST == 1:
        XPROB[0] = 1.0

    # schedule event for next end of year
    ATRIB[0] = CLOCK + 8760
    ATRIB[1] = 4
    NUMINQ, MFA, IPOINT = filem(MFA, ATRIB, NUMINQ, IPOINT, EVNTS)

    IYEAR = (CLOCK+1)/8760
    XYEAR = float(IYEAR)
    JDAY = 0
    
    for N in range(NFCST):
        for IAR in range(NOAREA):
            # COMPUTE LOLES, SUM FOR FINAL REPORT, TAKE WEIGHTED AVG

            # AREA LOLE FOR FORECAST N IS SUM OF TRANS & GEN LOLES
            LOLSHA[IAR,N] = LOLTHA[IAR,N] + LOLGHA[IAR,N]
            # CUMULATIVE SUM OF HOURLY AREA LOLES FOR THIS FORECAST IS NEXT
            SOLSHA[IAR,N] = SOLSHA[IAR,N] + float(LOLSHA[IAR,N])
            # CUMULATIVE SUM OF HOURLY AREA LOLES ASSIGNED TO TRANSMISSION
            SOLTHA[IAR,N] = SOLTHA(IAR,N) + float(LOLTHA(IAR,N))
            SOLGHA[IAR,N] = SOLGHA(IAR,N) + float(LOLGHA(IAR,N))
            WOLSHA[IAR] = WOLSHA(IAR) + float(LOLSHA(IAR,N))*XPROB(N)
            WOLGHA[IAR] = WOLGHA(IAR) + float(LOLGHA(IAR,N))*XPROB(N)
            WOLTHA[IAR] = WOLTHA(IAR) + float(LOLTHA(IAR,N))*XPROB(N)
            # COMPUTE TOTAL MAGNITUDES, SUM FOR FINAL REPORT, TAKE WEIGHTED AVG
            MGNSHA[IAR,N] = MGNTHA[IAR,N] + MGNGHA[IAR,N]
            SGNSHA[IAR,N] = SGNSHA[IAR,N] + float(MGNSHA[IAR,N])
            SGNTHA[IAR,N] = SGNTHA[IAR,N] + float(MGNTHA[IAR,N])
            SGNGHA[IAR,N] = SGNGHA[IAR,N] + float(MGNGHA[IAR,N])
            WGNSHA[IAR] = WGNSHA[IAR] + float(MGNSHA[IAR,N])*XPROB[N]
            WGNGHA[IAR] = WGNGHA[IAR] + float(MGNGHA[IAR,N])*XPROB[N]
            WGNTHA[IAR] = WGNTHA[IAR] + float(MGNTHA[IAR,N])*XPROB[N]            
            
    for IAR in range(NOAREA):
        SWNGHA[IAR] = SWNGHA[IAR] + WGNGHA[IAR]
        SWNTHA[IAR] = SWNTHA[IAR] + WGNTHA[IAR]
        SWNSHA[IAR] = SWNSHA[IAR] + WGNSHA[IAR]
        SWLGHA[IAR] = SWLGHA[IAR] + WOLGHA[IAR]
        SWLTHA[IAR] = SWLTHA[IAR] + WOLTHA[IAR]
        SWLSHA[IAR] = SWLSHA[IAR] + WOLSHA[IAR]                 
            
    for N in range(NFCST):        
        for IAR in range(NOAREA):
            # COMPUTE LOLES, SUM FOR FINAL REPORT, TAKE WEIGHTED AVG
            LOLSPA(IAR,N) = LOLTPA(IAR,N) + LOLGPA(IAR,N)
            SOLSPA(IAR,N) = SOLSPA(IAR,N) + float(LOLSPA(IAR,N))
            SOLTPA(IAR,N) = SOLTPA(IAR,N) + float(LOLTPA(IAR,N))
            SOLGPA(IAR,N) = SOLGPA(IAR,N) + float(LOLGPA(IAR,N))
            WOLGPA(IAR) = WOLGPA(IAR) + float(LOLGPA(IAR,N))*XPROB(N)
            WOLTPA(IAR) = WOLTPA(IAR) + float(LOLTPA(IAR,N))*XPROB(N)
            WOLSPA(IAR) = WOLSPA(IAR) + float(LOLSPA(IAR,N))*XPROB(N)
            
            # COMPUTE TOTAL MAGNITUDES, SUM FOR FINAL REPORT, TAKE WEIGHTED AVG
            MGNSPA(IAR,N) = MGNTPA(IAR,N) + MGNGPA(IAR,N)
            SGNSPA(IAR,N) = SGNSPA(IAR,N) + float(MGNSPA(IAR,N))
            SGNTPA(IAR,N) = SGNTPA(IAR,N) + float(MGNTPA(IAR,N))
            SGNGPA(IAR,N) = SGNGPA(IAR,N) + float(MGNGPA(IAR,N))
            WGNGPA(IAR) = WGNGPA(IAR) + float(MGNGPA(IAR,N))*XPROB(N)
            WGNTPA(IAR) = WGNTPA(IAR) + float(MGNTPA(IAR,N))*XPROB(N)
            WGNSPA(IAR) = WGNSPA(IAR) + float(MGNSPA(IAR,N))*XPROB(N)      


    # POOL STATISTICS, TOTAL, CUMULATE, WEIGHTED AVERAGE
    for IAR in range(NOAREA):
        SWNGPA(IAR) = SWNGPA(IAR) + WGNGPA(IAR)
        SWNTPA(IAR) = SWNTPA(IAR) + WGNTPA(IAR)
        SWNSPA(IAR) = SWNSPA(IAR) + WGNSPA(IAR)
        SWLGPA(IAR) = SWLGPA(IAR) + WOLGPA(IAR)
        SWLTPA(IAR) = SWLTPA(IAR) + WOLTPA(IAR)
        SWLSPA(IAR) = SWLSPA(IAR) + WOLSPA(IAR)
   
    NOERR=NORR
    for IAR in range(NOAREA):
        IPHOUR = LOLSHA(IAR,NOERR) + 1
        if IPHOUR > 22:
            IPHOUR = 22
        IPDP = LOLSPA(IAR,NOERR) + 1
        if IPDP > 22:
            IPDP = 22
        IPEUE = MGNSHA(IAR,NOERR)/LSTEP + 1
        if IPEUE > 22:
            IPEUE = 22
        HLOLE(IAR,IPHOUR) += 1.0
        DPLOLE(IAR,IPDP) += 1.0
        EUES(IAR,IPEUE) += 1.0

    for N in range(NFCST):
        # COMPUTE LOLES, SUM FOR FINAL REPORT, TAKE WEIGHTED AVG
	    LOLSHP(N) = LOLTHP(N) + LOLGHP(N)
	    SOLSHP(N) = SOLSHP(N) + float(LOLSHP(N))
	    SOLTHP(N) = SOLTHP(N) + float(LOLTHP(N))
	    SOLGHP(N) = SOLGHP(N) + float(LOLGHP(N))
	    WOLGHP = WOLGHP + float(LOLGHP(N))*XPROB(N)
	    WOLTHP = WOLTHP + float(LOLTHP(N))*XPROB(N)
	    WOLSHP = WOLSHP + float(LOLSHP(N))*XPROB(N)
        
        # COMPUTE TOTAL MAGNITUDES, SUM FOR FINAL REPORT, TAKE WEIGHTED AVG
	    MGNSHP(N) = MGNTHP(N) + MGNGHP(N)
	    SGNSHP(N) = SGNSHP(N) + float(MGNSHP(N))
	    SGNTHP(N) = SGNTHP(N) + float(MGNTHP(N))
	    SGNGHP(N) = SGNGHP(N) + float(MGNGHP(N))
	    WGNGHP = WGNGHP + float(MGNGHP(N))*XPROB(N)
	    WGNTHP = WGNTHP + float(MGNTHP(N))*XPROB(N)
	    WGNSHP = WGNSHP + float(MGNSHP(N))*XPROB(N)

    SWNGHP += WGNGHP
    SWNTHP += WGNTHP
    SWNSHP += WGNSHP
    SWLGHP += WOLGHP
    SWLTHP += WOLTHP
    SWLSHP += WOLSHP

    for N in range(NFCST):
        # COMPUTE LOLES, SUM FOR FINAL REPORT, TAKE WEIGHTED AVG
	    LOLSPP(N) = LOLTPP(N) + LOLGPP(N)
	    SOLSPP(N) = SOLSPP(N) + float(LOLSPP(N))
	    SOLTPP(N) = SOLTPP(N) + float(LOLTPP(N))
	    SOLGPP(N) = SOLGPP(N) + float(LOLGPP(N))
	    WOLGPP = WOLGPP + float(LOLGPP(N))*XPROB(N)
	    WOLTPP = WOLTPP + float(LOLTPP(N))*XPROB(N)
	    WOLSPP = WOLSPP + float(LOLSPP(N))*XPROB(N)
        # COMPUTE TOTAL MAGNITUDES, SUM FOR FINAL REPORT, TAKE WEIGHTED AVG
	    MGNSPP(N) = MGNTPP(N) + MGNGPP(N)
	    SGNSPP(N) = SGNSPP(N) + float(MGNSPP(N))
	    SGNTPP(N) = SGNTPP(N) + float(MGNTPP(N))
	    SGNGPP(N) = SGNGPP(N) + float(MGNGPP(N))
	    WGNGPP = WGNGPP + float(MGNGPP(N))*XPROB(N)
	    WGNTPP = WGNTPP + float(MGNTPP(N))*XPROB(N)
	    WGNSPP = WGNSPP + float(MGNSPP(N))*XPROB(N)
    
    SWNGPP = SWNGPP + WGNGPP
    SWNTPP = SWNTPP + WGNTPP
    SWNSPP = SWNSPP + WGNSPP
    SWLGPP = SWLGPP + WOLGPP
    SWLTPP = SWLTPP + WOLTPP
    SWLSPP = SWLSPP + WOLSPP

    # COMPUTE SQUARE OF VARIABLES
    for IAR in range(NOAREA):
        XNEWA(IAR,1)=XNEWA(IAR,1)+(WOLSHA(IAR))**2
        XNEWA(IAR,2)=XNEWA(IAR,2)+(WGNSHA(IAR))**2
        XNEWA(IAR,3)=XNEWA(IAR,3)+(WOLSPA(IAR))**2
        
    XNEWP(1)=XNEWP(1)+WOLSHP**2
    XNEWP(2)=XNEWP(2)+WGNSHP**2
    XNEWP(3)=XNEWP(3)+WOLSPP**2

#  printing part:
# PRINT ANNUALS & RESET TO 0  
    f = open("output.txt", "a") # can put earlier
      
    if IOJ != 0:
        f.write(" \n ")
        ITAB=ITAB+1
        f.write(" \n TABLE \n ")
        f.write(" \n RESULTS AFTER %d REPLICATIONS \n", IYEAR)
        
        if INDX != 1: # maybe 0, need to check later
            f.write(" \n  AREA FORECAST    HOURLY STATISTICS     PEAK STATISTICS    REMARKS\n")
            f.write(" \n NO   NO         HLOLE    XLOL     EUE       LOLE       XLOL\n")
            f.write("\n                (HRS/YR)   (MW)      (MWH)     (DAYS/YR)  (MW)\n")
        else:
            f.write("\n  AREA FORECAST    PEAK STATISTICS            REMARKS\n")
            f.write("\n  NO   NO          LOLE       XLOL\n")
            f.write("\n                   (DAYS/YR)   MW\n")

        for j in range(NOAREA):
            for N in range(nfcst):
                if LOLGHA(J,N) > 0:
                    XMGNA = float(MGNGHA(J,N))/float(LOLGHA(J,N))
                else:
                    XMGNA = 0.0
	            
                if LOLGPA(J,N) > 0:
                    XMGNP = float(MGNGPA(J,N))/float(LOLGPA(J,N))
                else:
        	        XMGNP = 0.0
        	      
                if INDX != 1:
                    f.write("\n  %2d  %2d    %4d    %8.2f  %8d     %4d     %8.2f            %s\n", 
                            J, N,LOLGHA(J,N),XMGNA,MGNGHA(J,N),LOLGPA(J,N),XMGNP,XG)
                    break
                else:
                    f.write("\n  %2d  %2d         %8.2f  %8.2f            %s\n", J,N,LOLGPA(J,N),XMGNP,XG)
        	      
                if LOLTHA(J,N) > 0:
        	        XMGNA = float(MGNTHA(J,N))/float(LOLTHA(J,N))
                else:
                    XMGNA = 0.0
                
                if LOLTPA(J,N) > 0:
        	        XMGNP = float(MGNTPA(J,N))/float(LOLTPA(J,N))
                else:
        	        XMGNP = 0.0
        		  
                if INDX != 1:
                    f.write("\n  %2d  %2d    %4d    %8.2f  %8d     %4d     %8.2f            %s\n",
                            J, N,LOLTHA(J,N),XMGNA,MGNTHA(J,N),LOLTPA(J,N),XMGNP,XT)
                    break
                else:
                    f.write("\n  %2d  %2d         %8.2f  %8.2f            %s\n", J,N,LOLTPA(J,N),XMGNP,XT)
         
                if LOLSHA(J,N) > 0:
                    XMGNA = float(MGNSHA(J,N))/float(LOLSHA(J,N))
                else:
        	        XMGNA = 0.0
        		  
                if LOLSPA(J,N) > 0:
                    XMGNP = float(MGNSPA(J,N))/float(LOLSPA(J,N))
                else:
        	        XMGNP = 0.0
                
                if INDX != 1:
                    f.write("\n  %2d  %2d    %4d    %8.2f  %8d     %4d     %8.2f            %s\n",
                            J,N,LOLSHA(J,N),XMGNA,MGNSHA(J,N),LOLSPA(J,N),XMGNP,XS)
                    break
                else:
                    f.write("\n  %2d  %2d         %8.2f  %8.2f            %s\n", J,N,LOLSPA(J,N),XMGNP,XT)
         	    
            if WOLGHA[j] > 0.0:
                XMGNA = WGNGHA(J)/WOLGHA(J)
            else:
                XMGNA = 0.0
    	    
            if WOLGPA[j] > 0.0:
                XMGNP = WGNGPA(J)/WOLGPA(J)
            else:
                XMGNP = 0.0
    	    
            if INDX != 1:
                f.write("\n  %2d  %s     %7.2f  %8.2f   %7.0f     %7.2f  %8.2f            %s\n", 
                        J,XA,WOLGHA(J),XMGNA,WGNGHA(J),WOLGPA(J),XMGNP,XG)
                break
            else:
                f.write("\n  %2d  %s         %8.2f  %8.2f            %s\n", J,XA,WOLGPA(J),XMGNP,XG)
    	    
            if WOLTHA(J) > 0.0:
                XMGNA = WGNTHA(J)/WOLTHA(J)
            else:
                XMGNA = 0.0
    	    
            if WOLTPA(J) > 0.0:
                XMGNP = WGNTPA(J)/WOLTPA(J)
            else:
                XMGNP = 0.0
    	    
            if INDX != 1:
                f.write("\n  %2d  %s     %7.2f  %8.2f   %7.0f     %7.2f  %8.2f            %s\n", 
                        J,XA,WOLTHA(J),XMGNA,WGNTHA(J) ,WOLTPA(J),XMGNP,XT)
                break
            else:
                f.write("\n  %2d  %s         %8.2f  %8.2f            %s\n", J,XA,WOLTPA(J),XMGNP,XT)
    	    
            if WOLSHA(J) > 0.0:
                XMGNA = WGNSHA(J)/WOLSHA(J)
            else:
                XMGNA = 0.0
    	    
            if WOLSPA(J) > 0.0:
                XMGNP = WGNSPA(J)/WOLSPA(J)
            else:
                XMGNP = 0.0
          
            if INDX != 1:
                f.write("\n  %2d  %s     %7.2f  %8.2f   %7.0f     %7.2f  %8.2f            %s\n", 
                        J,XA,WOLSHA(J),XMGNA,WGNSHA(J),WOLSPA(J),XMGNP,XS)
                break
            else:
                f.write("\n  %2d  %s         %8.2f  %8.2f            %s\n", J,XA,WOLSPA(J),XMGNP,XS)
        
                   
        # C POOL STATISTICS
        f.write("\n  POOL STATISTICS \n")
        for N in range(NFCST):
            if LOLGHP(N)> 0:
                XMGNH = float(MGNGHP(N))/float(LOLGHP(N))
            else:
                XMGN = 0.0
            
            if LOLGPP(N) > 0:
                XMGNP = float(MGNGPP(N))/float(LOLGPP(N))
            else:
                XMGN = 0.0

            if INDX != 1:
                WRITE(16,1031) N,LOLGHP(N),XMGNH,MGNGHP(N)
                *,LOLGPP(N),XMGNP,XG
                break
            else:
                WRITE(16,2031)N,LOLGPP(N),XMGNP,XG
    	    
    		if LOLTHP(N) > 0:
                XMGNH = float(MGNTHP(N))/float(LOLTHP(N))
    	    else:
                XMGNH = 0.0
    	    
    		if LOLTPP(N) > 0:
                XMGNP= float(MGNTPP(N))/float(LOLTPP(N))
    	    else:
                XMGNP= 0.0
    	    
            if INDX != 1:
               WRITE(16,1031) N,LOLTHP(N),XMGNH,MGNTHP(N)
               *,LOLTPP(N),XMGNP,XT
                break
            else:
                WRITE(16,2031) N,LOLTPP(N),XMGNP,XT
    	
    	    if LOLSHP(N) > 0:
                XMGNH= float(MGNSHP(N))/float(LOLSHP(N))
    	    else:
                XMGNH= 0.0
    	    
    	    if LOLSPP(N) > 0:
                XMGNP= float(MGNSPP(N))/float(LOLSPP(N))
    	    else:
                XMGNP= 0.0
                
    	    if INDX != 1:
                WRITE(16,1031) N,LOLSHP(N),XMGNH,MGNSHP(N)
                *,LOLSPP(N),XMGNP,XS
                break
            else:
                WRITE(16,2031)N,LOLSPP(N),XMGNP,XS          
                
        if WOLGHP > 0.0:
            XMGNH= WGNGHP/WOLGHP
        else:
            XMGNH= 0.0
        
        if WOLGPP > 0.0:
            XMGNP= WGNGPP/WOLGPP
        else:
            XMGNP= 0.0
        
        if INDX != 1:
            WRITE(16,1041)XA,WOLGHP,XMGNH,WGNGHP
            *,WOLGPP,XMGNP,XG
            break
        else:
            WRITE(16,2041)XA,WOLGPP,XMGNP,XG
 
    	if WOLTHP > 0.0:
            XMGNH= WGNTHP/WOLTHP
        else:
            XMGNH= 0.0
	 
    	if WOLTPP > 0.0:
            XMGNP= WGNTPP/WOLTPP
        else:
            XMGNP= 0.0

    	if INDX != 1:
            WRITE(16,1041)XA,WOLTHP,XMGNH,WGNTHP
             *,WOLTPP,XMGNP,XT
            break
        else:
            WRITE(16,2041)XA,WOLTPP,XMGNP,XT
 
    	if WOLSHP > 0.0:
            XMGNH= WGNSHP/WOLSHP
    	else:
            XMGNH= 0.0
    	 
        if WOLSPP > 0.0:
            XMGNP= WGNSPP/WOLSPP
        else:
            XMGNP= 0.0
             
        if INDX != 1:
            WRITE(16,1041)XA,WOLSHP,XMGNH,WGNSHP
             *,WOLSPP,XMGNP,XS
             break
        else:
            WRITE(16,2041)XA, WOLSPP,XMGNP,XS 
    
    f.close() # close the file "output"
    
    # zero out statistics reated arrays and scalars:
    for IAR in range(NOAREA):
         WGNGHA(IAR)=0.0
         WGNTHA(IAR)=0.0
         WGNSHA(IAR)=0.0
         WOLGHA(IAR)=0.0
         WOLTHA(IAR)=0.0
         WOLSHA(IAR)=0.0    
    
    for IAR in range(NOAREA):
         WGNGPA(IAR)=0.0
         WGNTPA(IAR)=0.0
         WGNSPA(IAR)=0.0
         WOLGPA(IAR)=0.0
         WOLTPA(IAR)=0.0
         WOLSPA(IAR)=0.0    
    
    for IAR in range(NOAREA):
         WGNSPA(IAR)=0.0
         WOLSPA(IAR)=0.0 
     
    WGNSHP=0.0
    WOLSHP=0.0
    WGNSPP=0.0
    WOLSPP=0.0
    WGNGHP=0.0
    WOLGHP=0.0
    WGNGPP=0.0
    WOLGPP=0.0
    WGNTHP=0.0
    WOLTHP=0.0
    WGNTPP=0.0
    WOLTPP=0.0        
    
    for N in range(NFCST):
        for IAR in range(NOAREA):
            LOLTHA[IAR,N] = 0
            LOLGHA[IAR,N] = 0
            MGNTHA[IAR,N] = 0
            MGNGHA[IAR,N] = 0
            LOLTPA[IAR,N] = 0
            LOLGPA[IAR,N] = 0
            MGNTPA[IAR,N] = 0
            MGNGPA[IAR,N] = 0
        LOLTHP[N] = 0
        LOLGHP[N] = 0
        MGNTHP[N] = 0
        MGNGHP[N] = 0
        LOLTPP[N] = 0
        LOLGPP[N] = 0
        MGNTPP[N] = 0
        MGNGPP[N] = 0  
    
    if IYEAR <= 5:
        WRITE(*,461) KWHERE, KVWHEN, KVSTAT, KVTYPE, KVLOC
        461 FORMAT(' ',' KVs = ',5I4)    
        
     
    # C BEGIN CHECKING FOR CONVERGENCE
    if KWHERE == 1:
        if KVWHEN == 1:
            if KVSTAT == 1: 
            # LOOP FOR HOURLY, AREA, LOLE
                if KVTYPE == 1:
                    SUM = SWLSHA[KVLOC]
                else:
                    SUM = SOLSHA[KVLOC,NOERR]
            # KVSTAT = 2, LOOP FOR HOURLY, AREA, EUE
            else:
                if KVTYPE == 1:
                    SUM = SWNSHA[KVLOC]
                else:
                    SUM = SGNSHA[KVLOC,NOERR]
        # KVWHEN = 2, LOOP FOR PEAK, AREA, LOLE
        else:
            if KVSTAT == 1:
                if KVTYPE == 1:
                    SUM = SWLSPA(KVLOC)
                else:
                    SUM = SOLSPA(KVLOC,NOERR)
            # KVWHEN=2, KVSTAT = 2, LOOP FOR PEAK, AREA, EUES
            else:
                if KVTYPE == 1:
                    SUM = SWNSPA(KVLOC)
                else:
                    SUM = SGNSPA(KVLOC,NOERR)          
    else:
        if KVWHEN == 1:
            if KVSTAT == 1:
            # LOOP FOR HOURLY, POOL, LOLE
                if KVTYPE == 1:
                    SUM = SWLSHP
                else:
                    SUM = SOLSHP(NOERR)
            # KVSTAT = 2, LOOP FOR HOURLY, POOL, EUE
            else:
                if KVTYPE == 1:
                    SUM = SWNSHP
                else:
                    SUM = SGNSHP(NOERR)
        # KVWHEN = 2, LOOP FOR PEAK, POOL, LOLE
        else:
            if KVSTAT == 1:
                if KVTYPE == 1:
                    SUM = SWLSPP
                else:
                    SUM = SOLSPP(NOERR)
            # KVWHEN=2, KVSTAT = 2, LOOP FOR PEAK, POOL, EUES
            else:
                if KVTYPE == 1:
                    SUM = SWNSPP
                else:
                    SUM = SGNSPP(NOERR)
    SUMX=SUM
    RFLAG, SSQ, XLAST = cvchk(CLOCK, FINISH, SUM, XLAST, SSQ, IYEAR, CVTEST):
    NOERR=1
    
    if IYEAR == INTVT:
        # CALL INTM
        intm()
        
    IFIN=FINISH/8760
    
    if IYEAR != INTVT:
        if IYEAR == IFIN:
            # CALL INTM
            intm()
    else:
        INTVT += INTV
        
    if RFLAG == 1:
        CALL report(IYEAR)
    else:
        if CLOCK >= FINISH:
          # CALL report(IYEAR)
          report(IYEAR)
          
    return          
                        
                    
           

    