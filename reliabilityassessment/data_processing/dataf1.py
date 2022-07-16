from reliabilityassessment.data_processing.input_processing import input_processing
from reliabilityassessment.data_processing.pind import pind
from reliabilityassessment.data_processing.readInputB import readInputB


def dataf1(filepath):
    """
    Data file reading and preprocessing

    :param list filepath: filepath to file 'INPUTB' and 'LEEI' respectively
    :return: (*tuple*)  -- the same return values as 'input_processing'
    """

    inputB_dict = readInputB(filepath[0])
    NAMU = [e[:5].strip("'") for e in inputB_dict["ZZUD"]["NAT"]]
    NUMP = [e[5:].strip("'") for e in inputB_dict["ZZUD"]["NAT"]]
    inputB_dict_nameToInt = pind(inputB_dict)

    return input_processing(inputB_dict_nameToInt, filepath[1], NAMU, NUMP)
