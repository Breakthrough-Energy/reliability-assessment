from hour import hour

import globalVar as g
from quartr import quartr
from week import week
from year import year


def events(number, RFLAG):
    """
    Simulation schedule for different time-scale of events

    :param: IERR int -- error indicator
    :param: number int -- Simulaiton type indicator
    :return: N/A
    """

    g.IERR = 0  # indicator/flag of run time error
    print("Events called with argument = %d" % (number))

    assert number in (1, 2, 3, 4), "The argument 'number' is invalid! Program aborted."

    if number == 1:
        # simulate hourly event
        hour()
        g.JHOUR = g.JHOURT
        return
    elif number == 2:
        # simulate weekly event
        week()
        return
    elif number == 3:
        # simulate quarterly event
        quartr()
        return
    else:  # number == 4:
        # simulate yearly event
        year(g.IERR, RFLAG)
        if g.IERR == 1:
            print("Error Iin subroutine year")
        return
