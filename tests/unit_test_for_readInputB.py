# -*- coding: utf-8 -*-

# Unit test for readInputB function
import numpy as np
from readInputB import readInputFileB

if __name__ =="__ main__":
    FileNameAndPath = 'INPUTB_for_unit_test'
    JSEED,NLS,IW1,IW2,IW3,KWHERE,KVWHEN,KVSTAT,KVTYPE,KVLOC,\
        CVTEST,FINISH,JSTEP,JFREQ,MAXEUE,IOI,IOJ,IREM,INTV,IREPD,IREPM,\
        SNRI,NAR,RATES,ID,\
        SNRI_ZZUD, NAT, NAR_ZZUD, HRLOAD, ID_ZZUD, \
        SNRI_ZZTD, ID_ZZTD, NAR_ZZTD, NAE, ADM, CAP, CAPR, PROP = readInputFileB(FileNameAndPath)
    
    # assertion for values in ZZMC data card
    assert JSEED == 345237
    assert NLS == 1
    assert IW1 == 13
    assert IW2 == 26
    assert IW3 == 39
    assert KWHERE == 1
    assert KVWHEN == 1
    assert KVSTAT == 1
    assert KVTYPE == 2
    assert KVLOC == 1
    assert CVTEST == 0.025
    assert FINISH == 9999
    assert JSTEP == 1
    assert JFREQ == 1
    assert MAXEUE == 1000
    assert IOI == 0
    assert IOJ == 0
    assert IREM == 1
    assert INTV == 5
    assert IREPD == 1
    assert IREPM == 1
    
    # assertion for values in ZZLD data card
    np.testing.assert_array_equal(SNRI,[1,2])
    np.testing.assert_array_equal(NAR,["'A1'", "'A2'"])
    np.testing.assert_array_equal(RATES[:,0],[3000, 3000])
    np.testing.assert_array_equal(ID[:,0],[1, 1])
    np.testing.assert_array_equal(ID[:,1],[52, 52])
    np.testing.assert_array_equal(ID[:,2],[31, 31])
    np.testing.assert_array_equal(ID[:,3],[32, 32])
    
    # assertion for values in ZZUD data card
    np.testing.assert_array_equal(SNRI_ZZUD,[1,2])
    np.testing.assert_array_equal(NAT,["'A10101'", "'A20101'"])
    np.testing.assert_array_equal(NAR_ZZUD,["'A1'", "'A2'"])
    np.testing.assert_array_equal(HRLOAD[:,0], [12.  , 12.  , 12.  , 12.  ,  0.  ,  0.02,  0.  ])
    np.testing.assert_array_equal(HRLOAD[:,1], [12.  , 12.  , 12.  , 12.  ,  0.  ,  0.02,  0.  ])
    np.testing.assert_array_equal(ID_ZZUD[0,:], [0., 0., 0., 0., 0.])
    np.testing.assert_array_equal(ID_ZZUD[1,:], [0., 0., 0., 0., 0.])
    
    # assertion for values in ZZTD data card
    np.testing.assert_array_equal(SNRI_ZZTD,[1])
    np.testing.assert_array_equal(ID_ZZTD,[1])
    np.testing.assert_array_equal(NAR_ZZTD[0],["'A1'"])
    np.testing.assert_array_equal(NAE[0],["'A2'"])
    np.testing.assert_array_equal(ADM[0,:],[-120.,  -60.,    0.,  -80.,  -40.,  -20.])
    np.testing.assert_array_equal(CAP[0,:],[300., 150.,   0., 150., 100.,  50.])
    np.testing.assert_array_equal(CAPR[0,:],[300., 150.,   0., 150., 100.,  50.])
    np.testing.assert_array_equal(PROP[0,:],[0.9216, 0.0768, 0.0016, 0., 0., 0.])
    
    
    
    
    
