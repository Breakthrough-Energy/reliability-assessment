from reliabilityassessment.monte_carlo.hour import hour
from reliabilityassessment.monte_carlo.quartr import quartr
from reliabilityassessment.monte_carlo.week import week
from reliabilityassessment.monte_carlo.year import year


def events(
    NUMBER,
    RFLAG,
    CLOCK,
    ATRIB,
    MFA,
    NUMINQ,
    IPOINT,
    EVNTS,
    PLNDST,
    JHOUR,
    JPLOUT,
    IQ,
    RATES,
    NHRSYR,
):
    """
    Simulation schedule for different time-scale of events

    :param: IERR int -- error indicator
    :param: NUMBER int -- Simulaiton type indicator
    :return: N/A
    """

    IERR = 0  # indicator/flag of run time error
    print("Events called with argument = %d" % (NUMBER))

    assert NUMBER in (1, 2, 3, 4), "The argument 'NUMBER' is invalid! Program aborted."

    if NUMBER == 1:
        # simulate hourly event
        JHOURT = hour()  # !the version of 'hour' is not complete
        JHOUR = JHOURT
        return
    elif NUMBER == 2:
        # simulate weekly event
        IPOINT, MFA, NUMINQ = week(
            CLOCK, ATRIB, MFA, NUMINQ, IPOINT, EVNTS, PLNDST, JHOUR, JPLOUT
        )
        return
    elif NUMBER == 3:
        # simulate quarterly event
        IQ, IPOINT, MFA, NUMINQ, RATING = quartr(
            IQ, RATES, CLOCK, ATRIB, NHRSYR, MFA, NUMINQ, IPOINT, EVNTS
        )
        return
    else:  # NUMBER == 4:
        # simulate yearly event
        IERR = year(IERR, RFLAG)  # !the version of 'year' is not complete
        if IERR == 1:
            print("Error in subroutine year")
        return
