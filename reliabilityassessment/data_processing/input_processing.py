from copy import deepcopy

import numpy as np

from reliabilityassessment.data_processing.datax import datax
from reliabilityassessment.data_processing.dpeak import dpeak
from reliabilityassessment.data_processing.plhr import plhr
from reliabilityassessment.data_processing.resca import resca
from reliabilityassessment.data_processing.smaint import smaint
from reliabilityassessment.data_processing.wpeakf import wpeakf
from reliabilityassessment.data_processing.xldnew import xldnew
from reliabilityassessment.data_processing.xporta import xporta
from reliabilityassessment.data_processing.xports import xports


def input_processing(inputB_dict_nameToInt, filePathToLEEI, NAMU, NUMP):

    """
    Applying furhter preprocessing logic on the loaded data and create/update certain arrays.

    :param dict inputB_dict_nameToInt: a compound python dictionary stores
                             all the parsed data from the INPUTB file.
    :param string filePathToLEEI: a file path to file 'LEEI'
    :param list NAMU: list of (string type) gen unit name
    :param list NUMP: list of (string type) gen plant name

    :return: (*tuple*)  a series of numpy ndarrays;
                        please refer to the 'variable description list.xlsx' file at
                        https://www.dropbox.com/s/eahg8x584s9pg4j/variable%20descriptions.xlsx?dl=0
    """

    (
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
        CAPCON,
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
        ID,
    ) = datax(inputB_dict_nameToInt)

    RATES = deepcopy(inputB_dict_nameToInt["ZZUD"]["HRLOAD"][0:4, :].T)

    NUNITS = len(CAPCON)
    # maybe we can explicitly input & return the varaible NUINTS?
    # since it is heavily used in many funcitons
    JPLOUT = np.zeros((52, NUNITS))

    # Read the hourly load data from the file 'LEEI'
    HRLOAD = xldnew(filePathToLEEI, PKLOAD)

    # Add interchanges
    xporta(JENT, INTCH, HRLOAD)

    # Find daily peaks and hour of daily peaks
    MAXHR, DYLOAD, MAXDAY = dpeak(HRLOAD, hour_within_day=24)

    # Finds weekly peaks (MW)
    WPEAK = wpeakf(DYLOAD)

    # Find hour of pool daily peaks
    MXPLHR = plhr(HRLOAD)

    # Remove interchanges and restore hrload
    xports(JENT, INTCH, HRLOAD)

    # Calculate the total owned cap and interchange at peak
    resca(SUSTAT[:, 1], MAXDAY, QTR, CAPOWN, RATES, JENT, INTCH)

    # Prepare jplout - table of planned outages of units
    # * return values are subjected to change *
    IREPM = inputB_dict_nameToInt["ZZMC"]["IREPM"]
    ITAB = 0
    JPLOUT, ITAB = smaint(
        NOAREA,
        ID,
        ITAB,
        RATES,
        PROBG,
        DERATE,
        PKLOAD,
        WPEAK,
        IREPM,
        MINRAN,
        MAXRAN,
        INHBT1,
        INHBT2,
        NAMU,
        NUMP,
    )

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
        CAPOWN,
        CAPCON,
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
        RATES,
        ID,
        HRLOAD,
        MAXHR,
        DYLOAD,
        MAXDAY,
        WPEAK,
        MXPLHR,
        JPLOUT,
        ITAB,
    )
