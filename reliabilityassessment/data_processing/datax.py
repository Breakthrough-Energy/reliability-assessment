from copy import deepcopy

import numpy as np


def datax(inputB_dict):
    """
    Reads all data except the hourly load data
    and creates appropriate arrays.

    :param dict inputB_dict: a compound python dictionary stores
                             all the parsed data from the INPUTB file.

    :return: (*tuple*)  a series of numpy ndarrays
    """

    # --------------------------------------------------------------
    # Read JSEED,LOSS SHARING POLICY,ENDING WEEKS OF FOUR SEASONS
    # and create QTR(3) -- HOURS OF THE YEAR FOR QUARTER CHANGE.
    # ISH = 0 # wariables used to indciate whether to wrtie some infor to TRAOUT file or not
    # '0' meanss 'not'
    QTR = np.zeros((3,)).astype("int")
    QTR[0] = inputB_dict["ZZMC"]["IW1"] * 168 + 0.5
    QTR[1] = inputB_dict["ZZMC"]["IW2"] * 168 + 0.5
    QTR[2] = inputB_dict["ZZMC"]["IW3"] * 168 + 0.5

    # --------------------------------------------------------------
    # Read or convert ZZLD data card:
    # area annual peak loads, area maintenance windows,
    # and constraint on the area sum of flows. Create PKLOAD(J),
    # MINRAN(J),MAXRAN(J), INHBT1(J) AND INHBT2(J)
    NORR = 1
    NFCST = 1
    NOAREA = len(inputB_dict["ZZLD"]["RATES"])
    PKLOAD = deepcopy(
        inputB_dict["ZZLD"]["RATES"][:, 0]
    )  # Annual peak in the area, in MW.
    FU = deepcopy(inputB_dict["ZZLD"]["RATES"][:, 1])  # forecasting uncertaint
    MINRAN = deepcopy(
        inputB_dict["ZZLD"]["ID"][:, 0]
    )  # BEG WK: Beginning week of the outage window
    MAXRAN = deepcopy(
        inputB_dict["ZZLD"]["ID"][:, 1]
    )  # END WK: Beginning week of the outage window
    INHBT1 = deepcopy(
        inputB_dict["ZZLD"]["ID"][:, 2]
    )  # BEG WK: Beginning week of the Forbidden Period.
    INHBT2 = deepcopy(
        inputB_dict["ZZLD"]["ID"][:, 3]
    )  # END WK: Beginning week of the Forbidden Period.

    BN = np.zeros((NOAREA, 5))
    BN[:, 3] = deepcopy(inputB_dict["ZZLD"]["RATES"][:, 2])  # Sum of Flows Constraint.
    BN[:, 4] = deepcopy(inputB_dict["ZZLD"]["RATES"][:, 2])  # Sum of Flows Constraint.

    for fu in FU:
        if fu != 0:
            NORR = 3
            NFCST = 5

    BN[:, 0] = [i for i in range(NOAREA)]
    BN[:, 1] = 0
    BN[:, 2] = 0

    SUSTAT = np.zeros((NOAREA, 6))
    SUSTAT[:, 0] = deepcopy(
        inputB_dict["ZZLD"]["RATES"][:, 0]
    )  # Annual peak in the area, in MW.

    FCTERR = np.zeros((NOAREA, 5))
    FCTERR[:, 2] = 1.0
    FCTERR[:, 3] = 1.0 - FU / 100.0
    FCTERR[:, 1] = 1.0 + FU / 100.0
    FCTERR[:, 0] = 1.0 + FU * 2.5 / 100.0
    FCTERR[:, 4] = 1.0 - FU * 2.5 / 100.0

    # Create PROBD array: prob dist of load forecast uncertainty.
    PROBD = np.array([0.067, 0.242, 0.382, 0.242, 0.067])

    # --------------------------------------------------------------
    # Read or convert ZZUD data card:
    # read unit data and create
    # DERATE(i): fraction of unit i available after derating
    # NOGEN(j): no. of generators in area j.
    # NUNITS: no. of generators in all areas.
    # PROB(k,i): cum. prob. that unit i is in state k.
    #                    k=1 full avail, k=2 derate
    # Rates(i,k): seasonal rating of unit i, k=1,4.
    #                   1= Jan, Mar 2=Apr, June, 3=July, Sept, 4=Oct, Dec.
    # NOGEN(j): number of generators in area j
    # NUNITS : number of generators in the entire model
    NUNITS = len(inputB_dict["ZZUD"])
    CAPOWN = np.zeros((NOAREA, NUNITS))
    CAPCON = np.zeros((NUNITS,))
    NOGEN = np.zeros((NOAREA,))
    PROBG = np.zeros((NUNITS, 2))
    DERATE = np.zeros((NUNITS,))
    for i in range(NUNITS):
        j = deepcopy(inputB_dict["ZZUD"]["NAR"])  # idx (int) for area
        j1 = deepcopy(inputB_dict["ZZUD"]["SNRI"])  # idx (int) for gen. unit
        CAPCON[j1] = j
        CAPOWN[j, j1] = 1.0
        NOGEN[j] = NOGEN[j] + 1
        PROBG[i, 0] = (
            1.0
            - inputB_dict["ZZUD"]["HRLOAD"][5, i]
            - inputB_dict["ZZUD"]["HRLOAD"][4, i]
        )
        PROBG[i, 1] = 1.0 - inputB_dict["ZZUD"]["HRLOAD"][5, i]
        DERATE[i] = 1.0 - inputB_dict["ZZUD"]["HRLOAD"][6, i] / 100.0

    # --------------------------------------------------------------
    # Read contract interchanges between areas and create :
    # JENT(J1,J2)-- matrix indicating areas which have interchange
    # If 0, none; if positive, the pointer L used in INTCH(L,N).
    # INTCH(L,N)- contract interchanges between areas indicated by pointer
    # L, day N.
    JENT = -1 * np.ones(
        (NOAREA, NOAREA), dtype=int
    )  # 60 is the maium posobel conut of fixed contracts
    INTCH = np.zeros((60, 365))  # 60 is the maium posobel conut of fixed contracts
    INTCHR = np.zeros((60, 2))  # 60 is the maium posobel conut of fixed contracts
    if inputB_dict["ZZFC"] and len(inputB_dict["ZZFC"]) > 0:
        IP = -1
        for i in range(len(inputB_dict["ZZFC"])):
            # (from area, to area, begin day, end day, MW)
            j1 = deepcopy(
                inputB_dict["ZZFC"]["NAR"]
            )  # idx (int) for from area (i.e., sending power)
            j2 = deepcopy(
                inputB_dict["ZZFC"]["NAE"]
            )  # idx (int) for to area (i.e., receiving power)
            if JENT[j1, j2] == -1:
                IP = IP + 1
                JENT[j1, j2] = IP
                INTCHR[IP, 0] = j1  # INTCHR : recording the from and to area no.
                INTCHR[IP, 1] = j2  # for the IP-th power transfer contract
            i1 = deepcopy(
                inputB_dict["ZZFC"]["BEG DAY"]
            )  # idx (int) for from area (i.e., sending power)
            i2 = deepcopy(
                inputB_dict["ZZFC"]["END DAY"]
            )  # idx (int) for to area (i.e., receiving power)
            for k in range(i1, i2):  # BEG DAY, END DAY
                INTCH[JENT[j1, j2], k] = deepcopy(inputB_dict["ZZFC"]["MW"])

    # --------------------------------------------------------------
    # Read ownership of units and create CAPOWN(j,i),
    # fraction of unit i owned by area j
    if inputB_dict["ZZOD"] and len(inputB_dict["ZZOD"]) > 0:
        for i in range(CAPOWN.shape[0]):  # CAPOWN.shape = (NOAREA, NUNITS)
            CAPOWN[i, :] = deepcopy(inputB_dict["ZZOD"]["PERCENT"])

    # --------------------------------------------------------------
    # Read line data and create
    # LINENO(j1,j2) line number between area j1 and j2.
    # NLINES   number of lines
    # PROBL(k,l) cumulative probability that line l is
    # in state k, k=1,...6
    NLINES = inputB_dict["ZZTD"].shape[0]
    LINENO = np.zeros((NOAREA, NOAREA))
    LP = np.zeros((NLINES, 3))
    LP = deepcopy(inputB_dict["ZZTD"]["LineID"])
    for lineIdx in range(NLINES):
        LP[lineIdx, 0] = deepcopy(
            inputB_dict["ZZTD"]["LineID"][lineIdx]
        )  # idx (int) for the line
        LP[lineIdx, 1] = deepcopy(
            inputB_dict["ZZTD"]["NAR"][lineIdx]
        )  # idx (int) for 'from area'
        LP[lineIdx, 2] = deepcopy(
            inputB_dict["ZZTD"]["NAE"][lineIdx]
        )  # idx (int) for 'to area'
        LINENO[LP[lineIdx, 1], LP[lineIdx, 2]] = LP[lineIdx, 0]

    # Convert discrete probability to cumulative probability
    # PROBL = inputB_dict["ZZTD"]["PROBT"].T # (6, NLINES)
    PROBL = deepcopy(inputB_dict["ZZTD"]["PROBT"].T)
    for lineIdx in range(NLINES):
        for m in range(1, 6):
            PROBL[m, lineIdx] = PROBL[m, lineIdx] + PROBL[m - 1, lineIdx]

    BLPA = np.zeros((NLINES, 18))
    for i in len(inputB_dict["ZZTD"]["ADM"]):
        if inputB_dict["ZZTD"]["ADM"][i] == 0:
            BLPA[i, 0:18:3] = -0.5
        else:
            BLPA[i, 0:18:3] = deepcopy(inputB_dict["ZZTD"]["ADM"][i])

        if inputB_dict["ZZTD"]["CAP"][i] == 0:
            BLPA[i, 1:18:3] = 1.0
        else:
            BLPA[i, 1:18:3] = deepcopy(inputB_dict["ZZTD"]["CAP"][i])

        if inputB_dict["ZZTD"]["CAPR"][i] == 0:
            BLPA[i, 2:18:3] = 1.0
        else:
            BLPA[i, 2:18:3] = deepcopy(inputB_dict["ZZTD"]["CAPR"][i])

    # Read critical unit data and create JCRIT(l) - for the l-th line:
    # position 1           line number
    # position 2           n =  number of critical units affecting this line
    # positions 3 to 2+n   critical units effecting this line
    # position  3+n        threshold capacity
    # positions 4+n,5+n    forward factors, backward factors (*100)
    MXCRIT = 0  # total (maximum) number of crittial units
    JCRIT = np.zeros((500,))  # critical (Affecting) unit id

    return (
        QTR,
        NORR,
        NFCST,
        NOAREA,
        PKLOAD,
        FU,
        MINRAN,
        MAXRAN,
        INHBT1,
        INHBT2,
        BN,
        SUSTAT,
        FCTERR,
        PROBD,
        CAPOWN,
        NOGEN,
        PROBG,
        DERATE,
        JENT,
        INTCH,
        INTCHR,
        LP,
        LINENO,
        PROBL,
        BLPA,
        MXCRIT,
        JCRIT,
    )
