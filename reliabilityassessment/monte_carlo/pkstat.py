def pkstat(NST, MARGIN, LOLTPP, LOLGPP, LOLTPA, MGNTPA, MGNTPP, LOLGPA, MGNGPA, MGNGPP):
    """
    Collect (daily) peak statistics at area level and pool level


    :param int NST: indicating the NST-th lood-forecasting level
    :param numpy.ndarray MARGIN: 1D array with shape (NOAREA, )
                                 the (orginal) generation margin of each area
                                 after deducting the load in that area
    :param numpy.ndarray LOLTPP to MGNGPP:
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

    NOAREA = len(MARGIN)
    SUM = 0.0

    for j in range(NOAREA):
        SUM += float(MARGIN[j])

    # Some area has a loss, else we wouldn't be here
    # Update pool peak LOLE
    if SUM >= 0:
        LOLTPP[NST] += 1
    else:
        LOLGPP[NST] += 1

    for j in range(NOAREA):
        X = MARGIN[j]
        if X < 0.0:
            if SUM > 0.0:
                LOLTPA[j, NST] = LOLTPA[j, NST] + 1
                MGNTPA[j, NST] = MGNTPA[j, NST] + int(-X)  # IFIX(-X)
                MGNTPP[NST] = MGNTPP[NST] + int(-X)  # IFIX(-X)
            else:
                LOLGPA[j, NST] = LOLGPA[j, NST] + 1
                MGNGPA[j, NST] = MGNGPA[j, NST] + int(-X)  # IFIX(-X)
                MGNGPP[NST] = MGNGPP[NST] + int(-X)  # IFIX(-X)
    return
