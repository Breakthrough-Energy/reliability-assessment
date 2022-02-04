def hrstat(
    NST, MARGIN, LOLTHP, LOLGHP, LSFLG, LOLTHA, MGNTHA, MGNTHP, LOLGHA, MGNGHA, MGNGHP
):
    """
    Collect hourly statistics at area level and pool level

    :param int NST: indicating the NST-th lood-forecasting level
    :param numpy.ndarray MARGIN: 1D array with shape (NOAREA, )
                                 the (orginal) generation margin of each area
                                 after deducting the load in that area
    :param numpy.ndarray LOLTHP to MGNGHP:
                    please refer to the following naming rules for those
                    6-digit statistic-related quantitites:
                    The first three digits identify the type of statistic
                        LOL means this is annual loss of load statistic of some type
                        MGN means this is annual sum of negative margins (EUE)
                        SOL means this is the cumulative sum, all years, LOLE
                        SNG means this is the cumulative sum, all years, EUE
                    The fourth digit identifies cause of outage
                        T is transmission
                        G is generation
                        S is sum of t & g
                     The fifth digit indicates whether the stat is hourly or peak
                         H is for hourly stats
                         P is for peak stats
                     The sixth digit is for area or pool
                         A is area
                         P is pool
                    Examples	:
                        SGNSPA  is cumulative sum, EUE, total of t&g, peak, area
                        LOLTHP  is annual loss of load, transmission, hourly, pool
    """

    SUM = 0.0

    NOAREA = MARGIN.shape[0]
    for j in range(NOAREA):
        SUM += float(MARGIN[j])

    # Somebody has a loss, else we wouldn't be here
    # Update pool peak LOLE
    if SUM >= 0.0:
        LOLTHP[NST] += 1
    else:
        LOLGHP[NST] += 1

    for j in range(NOAREA):
        X = MARGIN[j]
        if X < 0.0:
            LSFLG[j] += 1
            if SUM > 0.0:
                LOLTHA[j, NST] = +1
                MGNTHA[j, NST] += int(-X)
                MGNTHP[NST] += int(-X)
            else:
                LOLGHA[j, NST] += 1
                MGNGHA[j, NST] += int(-X)
                MGNGHP[NST] += int(-X)
    return
