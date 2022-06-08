import os.path

from reliabilityassessment.data_processing.dataf1 import dataf1
from reliabilityassessment.monte_carlo.contrl import contrl
from reliabilityassessment.monte_carlo.initfl import initfl
from reliabilityassessment.monte_carlo.initl import initl
from reliabilityassessment.monte_carlo.rstart import rstart


def narp2020():
    """
    Main entry function of the whole NARP program

    ..note: The whole logic of this function itself is simple,
            but some child functions are not finalized yet.
            Put it here to assist understanding of the overall logic.
    """

    print("\n NARP Version-XX,  MM/DD/YY \n")

    # if DUMP file exists, go to function rstart:
    IRST = 0
    if os.path.exists("DUMP.pkl"):
        IRST = 1

    NFCST = 1
    NOERR = 1
    LVLTRC = 0
    NHRSYR = 8760

    print("\n call function initfl \n")
    MFA, IPOINT, EVNTS = initfl()
    print("\n call function dataf1 \n")

    # CALL DATAF1(JU): TBD
    MAXEUE, JSTEP, FINISH, INTV, IREM = dataf1()

    LSTEP = MAXEUE / 20
    ATRIB, CLOCK, IPOINT, MFA, NUMINQ = initl(JSTEP, EVNTS)
    IFIN = FINISH / 8760

    INTVT = INTV
    if IRST != 0:
        snapshot_data = rstart("DUMP.pkl")
        INTVTT = CLOCK / 8760
        print("**           Warning ! This is a restart case!\n            ")
        print("**           NUMBER OF PREVIOUS REPLICATIONS : %8d \n", INTVTT)

    # No return value is the intened behavior for function 'contrl'
    # * Its input argument list is not finalized *
    RFLAG = 0
    contrl(
        RFLAG,
        CLOCK,
        IPOINT,
        EVNTS,
        ATRIB,
        FINISH,
        NFCST,
        NOERR,
        LVLTRC,
        NHRSYR,
        LSTEP,
        IFIN,
        INTVT,
        snapshot_data,
    )

    if IREM == 0:
        os.remove("DUMP.pkl")
