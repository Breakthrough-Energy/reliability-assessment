from reliabilityassessment.monte_carlo.events import events
from reliabilityassessment.monte_carlo.remove import remove
from reliabilityassessment.monte_carlo.report import report

# Special Note: The function upload here is for the help in understanding'filem' function
# This function itself is simple; but since multiple deep-level
# children function calls (triggered here by function 'event') are involved;
# Thus, it is NOT completely ready.
# DELETE THIS SPECIAL NOTE IN THE FUTURE WHEN THIS FUCNTION IS LATER FINALIZLED


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
     :param int RFLAG: ureport flag, indicating whether or not to generate a output report
                                     with all the so-far obtianed reliability assessment results

     :return: (*tuple*) -- TBD
    """

    print("In function contrl")

    if IPOINT == 0:
        print("Error: Pointer To Event File Is 0!")
        return

    if CLOCK > EVNTS[IPOINT + 1]:
        print("Error: Next Event To Occur Scheduled Prior To Current Time!")
        print("ATRIB vector is: ")
        print(ATRIB)
        return

    if RFLAG == 1.0:
        return

    if EVNTS[IPOINT + 1] > (FINISH + 1.0):
        IYEAR = CLOCK / 8760.0
        report(IYEAR)
        return
    else:
        remove()
        CLOCK = ATRIB(1)
        NUMBER = ATRIB(2)

        # event funciton is not ready; so has to mock at this moment
        events(NUMBER, RFLAG)

    return  # TBD; we might have explict return-values here
