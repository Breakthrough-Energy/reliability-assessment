from reliabilityassessment.monte_carlo.events import events
from reliabilityassessment.monte_carlo.remove import remove
from reliabilityassessment.monte_carlo.report import report


def contrl(RFLAG, CLOCK, IPOINT, EVNTS, ATRIB, FINISH):
    """
     Pulls The Next Event From File And Checks Time. Events Is Called.
     Unless End Of Simulation Is Reached, Then Report Is Called

    :param int MFA: pointer of first available (i.e. empty) entry
                     in the event list
     :param numpy.ndarray ATRIB: events attribute vector of length 2
                                 ATRIB[0] -- global simulation clock (in unit: hour)
                                 ATRIB[1] -- int, simulation type
                                             0: hourly
                                             1: weekly
                                             2: quarterly
                                             3: yearly
     :param int NUMINQ: up-to-date total number of event entries (inquires)
     :param int IPOINT:  pointer of the first (already/previously stored) entry
                         initial value = -1
     :param numpy.ndarray EVNTS: 1D array of events
     :param int RFLAG: report flag, indicating whether or not to generate a output report
                                     with all the so-far obtianed reliability assessment results
    """

    print("In function 'contrl': ")
    NUMINQ = 0  # initialize NUMINQ

    while True:

        if IPOINT == -1:  # 0 in original Fortran
            print("Error: Pointer to event list Is -1!")
            return

        if CLOCK > EVNTS[IPOINT + 1]:
            print("Error: next event to occur scheduled prior to current time!")
            print("ATRIB vector is: ")
            print(ATRIB)
            return

        if RFLAG == 1:
            return

        if EVNTS[IPOINT + 1] > (FINISH + 1.0):
            IYEAR = CLOCK // 8760
            report(IYEAR)  # No return values of 'report' are explitly needed here
            return
        else:
            MFA, NUMINQ, IPOINT = remove(NUMINQ, ATRIB, EVNTS, IPOINT)
            CLOCK = ATRIB[0]
            NUMBER = ATRIB[1]  # simulation type

            # Only partial return values of the 'events' funciton are needed here
            # ! Not a finalized verison yet !
            IPOINT, CLOCK, EVNTS, ATRIB, RFLAG, IYEAR, NUMINQ, *_ = events(
                NUMBER, RFLAG
            )
    return
