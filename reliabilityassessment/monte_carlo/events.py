from reliabilityassessment.monte_carlo.hour import hour
from reliabilityassessment.monte_carlo.quartr import quartr
from reliabilityassessment.monte_carlo.week import week
from reliabilityassessment.monte_carlo.year import year


def events(
    number,
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
    :param: number int -- Simulaiton type indicator
    :return: N/A
    """

    IERR = 0  # indicator/flag of run time error
    print("Events called with argument = %d" % (number))

    assert number in (1, 2, 3, 4), "The argument 'number' is invalid! Program aborted."

    if number == 1:
        # simulate hourly event
        JHOURT = hour()
        JHOUR = JHOURT
        return
    elif number == 2:
        # simulate weekly event
        IPOINT, MFA, NUMINQ = week(
            CLOCK, ATRIB, MFA, NUMINQ, IPOINT, EVNTS, PLNDST, JHOUR, JPLOUT
        )
        return
    elif number == 3:
        # simulate quarterly event
        IQ, IPOINT, MFA, NUMINQ, RATING = quartr(
            IQ, RATES, CLOCK, ATRIB, NHRSYR, MFA, NUMINQ, IPOINT, EVNTS
        )
        return
    else:  # number == 4:
        # simulate yearly event
        IERR = year(IERR, RFLAG)
        if IERR == 1:
            print("Error Iin subroutine year")
        return
