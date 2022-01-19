
def transm(IFLAG,JFLAG):
    # DOUBLE PRECISION BLP
    # REAL INJ,INJB,LOD,LODC,INJ1
    # DIMENSION BLP(100,3),BNS(20,6)       
    
    LCLOCK=CLOCK
    JHOUR=JHOUR
    IOI=IOI
    NN=NOAREA
    NL=NLINES
    
    if IFLAG != 1:
        for i in range(NL):
            BLP[i,0]=BLPA[i,0]
            BLP[i,1]=BLPA[i,1]
            BLP[i,2]=BLPA[i,2]
            
    # CALL TM1(BN,LP,BLP,BLP0,NN,NL,BB,ZB,LT,NR)
    BLP0, BB, LT, ZB = tm1(BN, LP, BLP, BLP0, NN, NL, BB, ZB, LT, NR)
    
    for i in range(NL):
        L=(LNSTAT(i)-1)*3
        BLP[i,0]=BLPA[i,L+1]
        BLP[i,1]=BLPA[i,L+2]*STMULT[i,0]
        BLP[i,2]=BLPA[i,L+3]*STMULT[i,1]
        
    if JFLAG != 1:
        NLST=NLS
        NLS=1
        for i in range(NN):
            BN[i,2]=TRNSFR[i]+TRNSFJ[i]
            BN[i,1]=0.
            BNS[i,0]=BN[i,2]
            BNS[i,1]=BN[i,1]
            BNS[i,4]=SYSCON[i]
            BNS[i,5]=CAPREQ[i]
            X=SYSCON[i]-CAPREQ[i]
            
            if X >= 0:
                if BN[i,2] < 0:
                    BN[i,2]=0.
            else:
                if X > BN[i,2]:
                    BN[i,2]=X

     #    CALL TM2(BN,LP,BLP,BLP0,NN,NL,BB,ZB,LT,NR,FLOW,NLS,BNS,LCLOCK
     # *,JHOUR,IOI,JFLAG,INDIC)
        TM2(BN,LP,BLP,BLP0,NN,NL,BB,ZB,LT,NR,FLOW,NLS,BNS,LCLOCK,JHOUR,IOI,JFLAG,INDIC)
        
        NLS=NLST
        KFLAG=1
        for i in range(NN):
            SADJ[i]=0.
            SADJ[i]=SYSCON[i]-CAPREQ[i]-BN[i,1]
            if SADJ[i] < 0:
                 KFLAG=0                    

        if KFLAG == 1:
            return
        
        JFLAG=1
        for i in range(NN):
            CADJ[i]=BN(i,1)*NLS
    
    if NLS != 0:
        for i in range(NN):
            BN[I,3]=SYSCON[I]-CAPREQ[I]-CADJ[I]
            BN[I,2]=CAPREQ[I]
            BNS[I,1]=BN[I,3]
            BNS[I,2]=BN[I,2]
            BNS[I,4]=BN[I,4]        
      
        for i in range(NL):
            J=LP[I,2]
            K=LP[I,3]
            BLP[I,2]=BLP[I,2]-FLOW[I]
            BLP[I,3]=BLP[I,3]+FLOW[I]
            if BLP[I,2] < 0.0:
                BLP[I,2]=0.
            if BLP[I,3] < 0.0:
                BLP[I,3]=0.
            BN[J,4]=BN[J,4]+FLOW[I]
            BN[J,5]=BN[J,5]-FLOW[I]
            BN[K,4]=BN[K,4]-FLOW[I] 
    else:
        for i in range(NN):
            BN[I,3]=SYSCON[I]-CAPREQ[I]-CADJ[I]
            BN[I,2]=CAPREQ[I]
            BNS[I,1]=BN[I,3]
            BNS[I,2]=BN[I,2]
            BNS[I,4]=BN[I,4]
   
    # CALL TM2(BN,LP,BLP,BLP0,NN,NL,BB,ZB,LT,NR,FLOW,NLS,BNS,LCLOCK
    #  *,JHOUR,IOI,JFLAG,INDIC)
    TM2(BN,LP,BLP,BLP0,NN,NL,BB,ZB,LT,NR,FLOW,NLS,BNS,LCLOCK,JHOUR,IOI,JFLAG,INDIC)
    
    for i in range(NN):
        BN[i,3]=BNS[i,3]
        BN[i,4]=BNS[i,3] # a typo ? 

    for i in range(NN):
        SADJ[i]=BN[i,1]-SYSCON[i]+CAPREQ[i]+CADJ[i]   
    
    return
        
        
        