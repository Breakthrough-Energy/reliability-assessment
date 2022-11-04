from reliabilityassessment.monte_carlo.events import events
from reliabilityassessment.monte_carlo.remove import remove
from reliabilityassessment.monte_carlo.report import report


def contrl(RFLAG, CLOCK, IPOINT, EVNTS, ATRIB, FINISH):
    """
     Pulls the next event from event list and checks its time stamp. Function ‘events’ is called.
     Unless the end of the whole simulation is reached, function ‘report’ is called

    :param int MFA: pointer of first available (i.e., empty) entry
                     in the event list
     :param numpy.ndarray ATRIB: events attribute vector of length 2
                                 ATRIB[0] -- global simulation clock (in unit: hour)
                                 ATRIB[1] -- int, simulation event type
                                             1: hourly
                                             2: weekly
                                             3: quarterly
                                             4: yearly
     :param int NUMINQ: up-to-date total number of event entries (inquires)
     :param int IPOINT:  pointer of the first (already/previously stored) entry
                         initial value = -1
     :param numpy.ndarray EVNTS: 1D array of events
     :param int RFLAG: report flag, indicating whether or not to generate an output report
                                     with all the so-far obtained reliability assessment results
    """

    print("In function 'contrl': ")
    NUMINQ = 0  # initialize NUMINQ

    while True:

        if IPOINT == -1:  # 0 in original Fortran
            print("Error: Pointer to event list Is -1!")
            return

        if CLOCK > EVNTS[IPOINT + 1]:
            print("Error: next event to occur is scheduled prior to the current time!")
            print(
                "ATRIB[0] (the time clock this event will occur (unit: hr)) is: %.2f"
                % (ATRIB[0])
            )
            print("ATRIB[0] (the type of this event) is: %d" % (ATRIB[1]))
            return

        if RFLAG == 1:
            return

        if EVNTS[IPOINT + 1] > (FINISH + 1.0):
            IYEAR = CLOCK // 8760
            report(IYEAR)  # No return values of 'report' are explicitly needed here
            return
        else:
            MFA, NUMINQ, IPOINT = remove(NUMINQ, ATRIB, EVNTS, IPOINT)
            CLOCK = ATRIB[0]
            NUMBER = int(ATRIB[1])  # simulation type

            # Only partial return values of the 'events' function are needed here
            # ! Not a finalized version yet!
            IPOINT, CLOCK, EVNTS, ATRIB, RFLAG, IYEAR, NUMINQ, *_ = events(
                NUMBER, RFLAG
            )
    return
