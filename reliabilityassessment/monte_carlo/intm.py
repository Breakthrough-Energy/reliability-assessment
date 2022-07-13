import pickle


def intm(
    ATRIB,
    CLOCK,
    IPOINT,
    XLAST,
    SSQ,
    MFA,
    NUMINQ,
    ITAB,
    INTVT,
    EVNTS,
    IGSEED,
    ILSEED,
    LT,
    BB,
    ZB,
    SOLTHA,
    SOLGHA,
    SOLSHA,
    SGNTHA,
    SGNGHA,
    SGNSHA,
    SOLTPA,
    SOLGPA,
    SOLSPA,
    SGNTPA,
    SGNGPA,
    SGNSPA,
    SWLSHA,
    SWLGHA,
    SWLTHA,
    SWNSHA,
    SWNGHA,
    SWNTHA,
    SWLSPA,
    SWLGPA,
    SWLTPA,
    SWNSPA,
    SWNGPA,
    SWNTPA,
    SOLTHP,
    SOLGHP,
    SOLSHP,
    SGNTHP,
    SGNGHP,
    SGNSHP,
    SOLTPP,
    SOLGPP,
    SOLSPP,
    SGNTPP,
    SGNGPP,
    SGNSPP,
    HLOLE,
    DPLOLE,
    EUES,
    XNEWA,
    XNEWP,
    WOLSHA,
    LSFLG,
):
    """
    Save intermediate results into a pickle file, DUMP.pkl, in case of unintentional or
    intentional interruption.

    .. note:: For descriptions of input variables, please refer to `variable
        descriptions.xlsx` in the project Dropbox folder.
    .. todo::
        1. Once the main loop of the sequential Monte Carlo simulation is
        implemented, refactor/remove this function according to the call signature
        such as directly save input variables into the caching file within the scope.
        2. If this function is kept during future refactor, update the doc strings of
        input variables according to the final version of the project doc.
        3. If this function is kept during future refactor, make the caching file
        name capable to be specified so that multiple users can run this module on a
        cluster with intermediate results caching due to interruptions.
    """

    snapshot_data = {
        "ATRIB": ATRIB[1],
        "CLOCK": CLOCK,
        "IPOINT": IPOINT,
        "XLAST": XLAST,
        "SSQ": SSQ,
        "MFA": MFA,
        "NUMINQ": NUMINQ,
        "ITAB": ITAB,
        "INTVT": INTVT,
        "EVNTS": EVNTS,
        "IGSEED": IGSEED,
        "ILSEED": ILSEED,
        "LT": LT,
        "BB": BB,
        "ZB": ZB,
        "SOLTHA": SOLTHA,
        "SOLGHA": SOLGHA,
        "SOLSHA": SOLSHA,
        "SGNTHA": SGNTHA,
        "SGNGHA": SGNGHA,
        "SGNSHA": SGNSHA,
        "SOLTPA": SOLTPA,
        "SOLGPA": SOLGPA,
        "SOLSPA": SOLSPA,
        "SGNTPA": SGNTPA,
        "SGNGPA": SGNGPA,
        "SGNSPA": SGNSPA,
        "SWLSHA": SWLSHA,
        "SWLGHA": SWLGHA,
        "SWLTHA": SWLTHA,
        "SWNSHA": SWNSHA,
        "SWNGHA": SWNGHA,
        "SWNTHA": SWNTHA,
        "SWLSPA": SWLSPA,
        "SWLGPA": SWLGPA,
        "SWLTPA": SWLTPA,
        "SWNSPA": SWNSPA,
        "SWNGPA": SWNGPA,
        "SWNTPA": SWNTPA,
        "SOLTHP": SOLTHP,
        "SOLGHP": SOLGHP,
        "SOLSHP": SOLSHP,
        "SGNTHP": SGNTHP,
        "SGNGHP": SGNGHP,
        "SGNSHP": SGNSHP,
        "SOLTPP": SOLTPP,
        "SOLGPP": SOLGPP,
        "SOLSPP": SOLSPP,
        "SGNTPP": SGNTPP,
        "SGNGPP": SGNGPP,
        "SGNSPP": SGNSPP,
        "HLOLE": HLOLE,
        "DPLOLE": DPLOLE,
        "EUES": EUES,
        "XNEWA": XNEWA,
        "XNEWP": XNEWP,
        "WOLSHA": WOLSHA,
        "LSFLG": LSFLG,
    }

    with open("DUMP.pkl", "wb") as fp:
        pickle.dump(snapshot_data, fp)