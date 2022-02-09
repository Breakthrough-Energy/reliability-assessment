
def flo(LP,BLP,FLOW,THET,SFLOW):
    """
    Computes line flows
    """
    
    i = 0
    while(True):
    
        j=LP[i,1]
        k=LP[i,2]
        
        FLOW[i]=THET[j]-THET[k]*BLP[i,0]
        SFLOW[k]=SFLOW[k]+FLOW[i]
        SFLOW[j]=SFLOW[j]-FLOW[i]
        
        i = i + 1
        if LP[i,0] == 0:
            break
    
    return # SFLOW; no need for explictly return since it is array (ref type)
    
        
