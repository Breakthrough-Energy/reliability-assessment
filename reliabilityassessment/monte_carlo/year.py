import numpy as np

from reliabilityassessment.monte_carlo.cvchk import cvchk
from reliabilityassessment.monte_carlo.filem import filem
from reliabilityassessment.monte_carlo.intm import intm
from reliabilityassessment.monte_carlo.report import report


def year(
    ATRIB,
    CLOCK,
    CVTEST,
    DPLOLE,
    EUES,
    EVNTS,
    FINISH,
    HLOLE,
    IPOINT,
    MFA,
    NUMINQ,
    RFLAG,
    LSFLG,
    NAMA,
    SSQ,
    SUSTAT,
    XLAST,
    BB,
    LT,
    ZB,
    IGSEED,
    ILSEED,
    INTV,
    INTVT,
    IOJ,
    KVLOC,
    KVSTAT,
    KVTYPE,
    KVWHEN,
    KWHERE,
    LSTEP,
    NFCST,
    NORR,
    FH,
    IERR,
    INDX,
    ITAB,
    IYEAR,
    SH,
    XA,
    XG,
    XS,
    XT,
    LOLGHA,
    LOLGHP,
    LOLGPA,
    LOLGPP,
    LOLSHA,
    LOLSHP,
    LOLSPA,
    LOLSPP,
    LOLTHA,
    LOLTHP,
    LOLTPA,
    LOLTPP,
    MGNGHA,
    MGNGHP,
    MGNGPA,
    MGNGPP,
    MGNSHA,
    MGNSHP,
    MGNSPA,
    MGNSPP,
    MGNTHA,
    MGNTHP,
    MGNTPA,
    MGNTPP,
    SGNGHA,
    SGNGHP,
    SGNGPA,
    SGNGPP,
    SGNSHA,
    SGNSHP,
    SGNSPA,
    SGNSPP,
    SGNTHA,
    SGNTHP,
    SGNTPA,
    SGNTPP,
    SOLGHA,
    SOLGHP,
    SOLGPA,
    SOLGPP,
    SOLSHA,
    SOLSHP,
    SOLSPA,
    SOLSPP,
    SOLTHA,
    SOLTHP,
    SOLTPA,
    SOLTPP,
    SWLGHA,
    SWLGHP,
    SWLGPA,
    SWLGPP,
    SWLSHA,
    SWLSHP,
    SWLSPA,
    SWLSPP,
    SWLTHA,
    SWLTHP,
    SWLTPA,
    SWLTPP,
    SWNGHA,
    SWNGHP,
    SWNGPA,
    SWNGPP,
    SWNSHA,
    SWNSHP,
    SWNSPA,
    SWNSPP,
    SWNTHA,
    SWNTHP,
    SWNTPA,
    SWNTPP,
    WGNGHA,
    WGNGHP,
    WGNGPA,
    WGNGPP,
    WGNSHA,
    WGNSHP,
    WGNSPA,
    WGNSPP,
    WGNTHA,
    WGNTHP,
    WGNTPA,
    WGNTPP,
    WOLGHA,
    WOLGHP,
    WOLGPA,
    WOLGPP,
    WOLSHA,
    WOLSHP,
    WOLSPA,
    WOLSPP,
    WOLTHA,
    WOLTHP,
    WOLTPA,
    WOLTPP,
):
    """
    Update related reliability statistics for yearly simulation

    .. note:: The inputs and outputs are massive scalars, 1D and 2D numpy.ndarrays.
             For descriptions of input and output variables, please refer to `variable descriptions.xlsx.`
             in the project Dropbox folder: https://www.dropbox.com/s/eahg8x584s9pg4j/variable%20descriptions.xlsx?dl=0

             For arrays LOLGHA to WOLTPP:
                    please refer to the following naming rules for those
                    6-digit statistic-related quantities:

                    The first three digits identify the type of statistic
                        LOL means this is an annual loss of load statistic of some type
                        MGN means this is the annual sum of negative margins (EUE)
                        SOL means this is the cumulative sum, all years, LOLE
                        SNG means this is the cumulative sum, all years, EUE
                    The fourth digit identifies the cause of the outage
                        T is transmission
                        G is generation
                        S is sum of t & g
                     The fifth digit indicates whether the stat is hourly or peak
                         H is for hourly stats
                         P is for peak stats
                     The sixth digit is for area or pool
                         A is area
                         P is pool

                    Examples	:
                        SGNSPA  is cumulative sum, EUE, total of t&g, peak, area
                        LOLTHP  is annual loss of load, transmission, hourly, pool
    """

    XPROB = np.zeros((5))

    if NFCST == 1:
        XPROB[0] = 1.0

    # Schedule event for next end of year
    ATRIB[0] = CLOCK + 8760
    ATRIB[1] = 4
    NUMINQ, MFA, IPOINT = filem(MFA, ATRIB, NUMINQ, IPOINT, EVNTS)

    IYEAR = (CLOCK + 1) / 8760
    NOAREA = EUES.shape[0]
    for N in range(NFCST):
        for IAR in range(NOAREA):
            # Compute LOLES, sum for final report, take weighted avg

            # Area LOLE for forecast N is sum of trans & gen LOLES
            LOLSHA[IAR, N] = LOLTHA[IAR, N] + LOLGHA[IAR, N]
            # Cumulative sum of hourly area LOLES for this forecast is next
            SOLSHA[IAR, N] += float(LOLSHA[IAR, N])
            # Cumulative sum of hourly area LOLES assigned to transmission
            SOLTHA[IAR, N] += float(LOLTHA[IAR, N])
            SOLGHA[IAR, N] += float(LOLGHA[IAR, N])
            WOLSHA[IAR] += float(LOLSHA[IAR, N]) * XPROB[N]
            WOLGHA[IAR] += float(LOLGHA[IAR, N]) * XPROB[N]
            WOLTHA[IAR] += float(LOLTHA[IAR, N]) * XPROB[N]
            # Compute total magnitudes, sum for final report, take weighted avg
            MGNSHA[IAR, N] = MGNTHA[IAR, N] + MGNGHA[IAR, N]
            SGNSHA[IAR, N] += float(MGNSHA[IAR, N])
            SGNTHA[IAR, N] += float(MGNTHA[IAR, N])
            SGNGHA[IAR, N] += float(MGNGHA[IAR, N])
            WGNSHA[IAR] += float(MGNSHA[IAR, N]) * XPROB[N]
            WGNGHA[IAR] += float(MGNGHA[IAR, N]) * XPROB[N]
            WGNTHA[IAR] += float(MGNTHA[IAR, N]) * XPROB[N]

    for IAR in range(NOAREA):
        SWNGHA[IAR] += WGNGHA[IAR]
        SWNTHA[IAR] += WGNTHA[IAR]
        SWNSHA[IAR] += WGNSHA[IAR]
        SWLGHA[IAR] += WOLGHA[IAR]
        SWLTHA[IAR] += WOLTHA[IAR]
        SWLSHA[IAR] += WOLSHA[IAR]

    for N in range(NFCST):
        for IAR in range(NOAREA):
            # Compute LOLES, sum for final report, take weighted avg
            LOLSPA[IAR, N] = LOLTPA[IAR, N] + LOLGPA[IAR, N]
            SOLSPA[IAR, N] += float(LOLSPA[IAR, N])
            SOLTPA[IAR, N] += float(LOLTPA[IAR, N])
            SOLGPA[IAR, N] += float(LOLGPA[IAR, N])
            WOLGPA[IAR] += float(LOLGPA[IAR, N]) * XPROB[N]
            WOLTPA[IAR] += float(LOLTPA[IAR, N]) * XPROB[N]
            WOLSPA[IAR] += float(LOLSPA[IAR, N]) * XPROB[N]

            # Compute total magnitudes, sum for final report, take weighted avg
            MGNSPA[IAR, N] = MGNTPA[IAR, N] + MGNGPA[IAR, N]
            SGNSPA[IAR, N] += float(MGNSPA[IAR, N])
            SGNTPA[IAR, N] += float(MGNTPA[IAR, N])
            SGNGPA[IAR, N] += float(MGNGPA[IAR, N])
            WGNGPA[IAR] += float(MGNGPA[IAR, N]) * XPROB[N]
            WGNTPA[IAR] += float(MGNTPA[IAR, N]) * XPROB[N]
            WGNSPA[IAR] += float(MGNSPA[IAR, N]) * XPROB[N]

    # POOL STATISTICS, TOTAL, CUMULATE, WEIGHTED AVERAGE
    for IAR in range(NOAREA):
        SWNGPA[IAR] += WGNGPA[IAR]
        SWNTPA[IAR] += WGNTPA[IAR]
        SWNSPA[IAR] += WGNSPA[IAR]
        SWLGPA[IAR] += WOLGPA[IAR]
        SWLTPA[IAR] += WOLTPA[IAR]
        SWLSPA[IAR] += WOLSPA[IAR]

    NOERR = NORR
    for IAR in range(NOAREA):
        IPHOUR = LOLSHA[IAR, NOERR] + 1
        if IPHOUR > 22:
            IPHOUR = 22
        IPDP = LOLSPA[IAR, NOERR] + 1
        if IPDP > 22:
            IPDP = 22
        IPEUE = MGNSHA[IAR, NOERR] / LSTEP + 1
        if IPEUE > 22:
            IPEUE = 22
        HLOLE[IAR, IPHOUR] += 1.0
        DPLOLE[IAR, IPDP] += 1.0
        EUES[IAR, IPEUE] += 1.0

    for N in range(NFCST):
        # Compute LOLES, sum for final report, take weighted avg
        LOLSHP[N] = LOLTHP[N] + LOLGHP[N]
        SOLSHP[N] += float(LOLSHP[N])
        SOLTHP[N] += float(LOLTHP[N])
        SOLGHP[N] += float(LOLGHP[N])
        WOLGHP += float(LOLGHP[N]) * XPROB[N]
        WOLTHP += float(LOLTHP[N]) * XPROB[N]
        WOLSHP += float(LOLSHP[N]) * XPROB[N]

        # Compute total magnitudes, sum for final report, take weighted avg
        MGNSHP[N] = MGNTHP[N] + MGNGHP[N]
        SGNSHP[N] += float(MGNSHP[N])
        SGNTHP[N] += float(MGNTHP[N])
        SGNGHP[N] += float(MGNGHP[N])
        WGNGHP += float(MGNGHP[N]) * XPROB[N]
        WGNTHP += float(MGNTHP[N]) * XPROB[N]
        WGNSHP += float(MGNSHP[N]) * XPROB[N]

    SWNGHP += WGNGHP
    SWNTHP += WGNTHP
    SWNSHP += WGNSHP
    SWLGHP += WOLGHP
    SWLTHP += WOLTHP
    SWLSHP += WOLSHP

    for N in range(NFCST):
        # Compute LOLES, sum for final report, take weighted avg
        LOLSPP[N] = LOLTPP[N] + LOLGPP[N]
        SOLSPP[N] += float(LOLSPP[N])
        SOLTPP[N] += float(LOLTPP[N])
        SOLGPP[N] += float(LOLGPP[N])
        WOLGPP += float(LOLGPP[N]) * XPROB[N]
        WOLTPP += float(LOLTPP[N]) * XPROB[N]
        WOLSPP += float(LOLSPP[N]) * XPROB[N]
        # Compute total magnitudes, sum for final report, take weighted avg
        MGNSPP[N] = MGNTPP[N] + MGNGPP[N]
        SGNSPP[N] += float(MGNSPP[N])
        SGNTPP[N] += float(MGNTPP[N])
        SGNGPP[N] += float(MGNGPP[N])
        WGNGPP += float(MGNGPP[N]) * XPROB[N]
        WGNTPP += float(MGNTPP[N]) * XPROB[N]
        WGNSPP += float(MGNSPP[N]) * XPROB[N]

    SWNGPP += WGNGPP
    SWNTPP += WGNTPP
    SWNSPP += WGNSPP
    SWLGPP += WOLGPP
    SWLTPP += WOLTPP
    SWLSPP += WOLSPP

    # Compute square of variables
    XNEWA = np.zeros((NOAREA, 3))
    XNEWP = np.zeros(3)

    for IAR in range(NOAREA):
        XNEWA[IAR, 0] += (WOLSHA[IAR]) ** 2
        XNEWA[IAR, 1] += (WGNSHA[IAR]) ** 2
        XNEWA[IAR, 2] += (WOLSPA[IAR]) ** 2

    XNEWP[0] += WOLSHP**2
    XNEWP[1] += WGNSHP**2
    XNEWP[2] += WOLSPP**2

    # Printing part:
    # Print result of annual analysis and reset related intermediate quantities Tto 0
    f = open("output.txt", "a")  # can put earlier

    if IOJ != 0:
        f.write(" \n ")
        ITAB += 1
        f.write(" \n TABLE \n ")
        f.write(" \n RESULTS AFTER %d REPLICATIONS \n" % (IYEAR))

        if INDX != 1:  # maybe 0, need to check later
            f.write(
                " \n  AREA FORECAST    HOURLY STATISTICS     PEAK STATISTICS    REMARKS\n"
            )
            f.write(" \n NO   NO         HLOLE    XLOL     EUE       LOLE       XLOL\n")
            f.write(
                "\n                (HRS/YR)   (MW)      (MWH)     (DAYS/YR)  (MW)\n"
            )
        else:
            f.write("\n  AREA FORECAST    PEAK STATISTICS            REMARKS\n")
            f.write("\n  NO   NO          LOLE       XLOL\n")
            f.write("\n                   (DAYS/YR)   MW\n")

        for J in range(NOAREA):
            for N in range(NFCST):
                if LOLGHA[J, N] > 0:
                    XMGNA = float(MGNGHA[J, N]) / float(LOLGHA[J, N])
                else:
                    XMGNA = 0.0

                if LOLGPA[J, N] > 0:
                    XMGNP = float(MGNGPA[J, N]) / float(LOLGPA[J, N])
                else:
                    XMGNP = 0.0

                if INDX != 1:
                    f.write(
                        "\n  %2d  %2d    %4d    %8.2f  %8d     %4d     %8.2f            %s\n"
                        % (
                            J,
                            N,
                            LOLGHA[J, N],
                            XMGNA,
                            MGNGHA[J, N],
                            LOLGPA[J, N],
                            XMGNP,
                            XG,
                        )
                    )
                else:
                    f.write(
                        "\n  %2d  %2d         %8.2f  %8.2f            %s\n"
                        % (J, N, LOLGPA[J, N], XMGNP, XG)
                    )

                if LOLTHA[J, N] > 0:
                    XMGNA = float(MGNTHA[J, N]) / float(LOLTHA[J, N])
                else:
                    XMGNA = 0.0

                if LOLTPA[J, N] > 0:
                    XMGNP = float(MGNTPA[J, N]) / float(LOLTPA[J, N])
                else:
                    XMGNP = 0.0

                if INDX != 1:
                    f.write(
                        "\n  %2d  %2d    %4d    %8.2f  %8d     %4d     %8.2f            %s\n"
                        % (
                            J,
                            N,
                            LOLTHA[J, N],
                            XMGNA,
                            MGNTHA[J, N],
                            LOLTPA[J, N],
                            XMGNP,
                            XT,
                        )
                    )
                else:
                    f.write(
                        "\n  %2d  %2d         %8.2f  %8.2f            %s\n"
                        % (J, N, LOLTPA[J, N], XMGNP, XT)
                    )

                if LOLSHA[J, N] > 0:
                    XMGNA = float(MGNSHA[J, N]) / float(LOLSHA[J, N])
                else:
                    XMGNA = 0.0

                if LOLSPA[J, N] > 0:
                    XMGNP = float(MGNSPA[J, N]) / float(LOLSPA[J, N])
                else:
                    XMGNP = 0.0

                if INDX != 1:
                    f.write(
                        "\n  %2d  %2d    %4d    %8.2f  %8d     %4d     %8.2f            %s\n"
                        % (
                            J,
                            N,
                            LOLSHA[J, N],
                            XMGNA,
                            MGNSHA[J, N],
                            LOLSPA[J, N],
                            XMGNP,
                            XS,
                        )
                    )
                else:
                    f.write(
                        "\n  %2d  %2d         %8.2f  %8.2f            %s\n"
                        % (J, N, LOLSPA[J, N], XMGNP, XT)
                    )

            if WOLGHA[J] > 0.0:
                XMGNA = WGNGHA[J] / WOLGHA[J]
            else:
                XMGNA = 0.0

            if WOLGPA[J] > 0.0:
                XMGNP = WGNGPA[J] / WOLGPA[J]
            else:
                XMGNP = 0.0

            if INDX != 1:
                f.write(
                    "\n  %2d  %s     %7.2f  %8.2f   %7.0f     %7.2f  %8.2f            %s\n"
                    % (J, XA, WOLGHA[J], XMGNA, WGNGHA[J], WOLGPA[J], XMGNP, XG)
                )
            else:
                f.write(
                    "\n  %2d  %s         %8.2f  %8.2f            %s\n"
                    % (J, XA, WOLGPA[J], XMGNP, XG)
                )

            if WOLTHA[J] > 0.0:
                XMGNA = WGNTHA[J] / WOLTHA[J]
            else:
                XMGNA = 0.0

            if WOLTPA[J] > 0.0:
                XMGNP = WGNTPA[J] / WOLTPA[J]
            else:
                XMGNP = 0.0

            if INDX != 1:
                f.write(
                    "\n  %2d  %s     %7.2f  %8.2f   %7.0f     %7.2f  %8.2f            %s\n"
                    % (J, XA, WOLTHA[J], XMGNA, WGNTHA[J], WOLTPA[J], XMGNP, XT)
                )
            else:
                f.write(
                    "\n  %2d  %s         %8.2f  %8.2f            %s\n"
                    % (J, XA, WOLTPA[J], XMGNP, XT)
                )

            if WOLSHA[J] > 0.0:
                XMGNA = WGNSHA[J] / WOLSHA[J]
            else:
                XMGNA = 0.0

            if WOLSPA[J] > 0.0:
                XMGNP = WGNSPA[J] / WOLSPA[J]
            else:
                XMGNP = 0.0

            if INDX != 1:
                f.write(
                    "\n  %2d  %s     %7.2f  %8.2f   %7.0f     %7.2f  %8.2f            %s\n"
                    % (J, XA, WOLSHA[J], XMGNA, WGNSHA[J], WOLSPA[J], XMGNP, XS)
                )
            else:
                f.write(
                    "\n  %2d  %s         %8.2f  %8.2f            %s\n"
                    % (J, XA, WOLSPA[J], XMGNP, XS)
                )

        # C POOL STATISTICS
        f.write("\n  POOL STATISTICS \n")
        for N in range(NFCST):
            if LOLGHP[N] > 0:
                XMGNH = float(MGNGHP[N]) / float(LOLGHP[N])
            # else:
            # XMGN = 0.0  # maybe for future usage

            if LOLGPP[N] > 0:
                XMGNP = float(MGNGPP[N]) / float(LOLGPP[N])
            # else:
            # XMGN = 0.0  # maybe for future usage

            if INDX != 1:
                f.write(
                    "      %2d     %4d     %8.2f  %8d     %4d     %8.2f            %s"
                    % (N, LOLGHP[N], XMGNH, MGNGHP[N], LOLGPP[N], XMGNP, XG)
                )
            else:
                f.write(
                    "      %2d         %8.2f  %8.2f            %s"
                    % (N, LOLGPP[N], XMGNP, XG)
                )

            if LOLTHP[N] > 0:
                XMGNH = float(MGNTHP[N]) / float(LOLTHP[N])
            else:
                XMGNH = 0.0

            if LOLTPP[N] > 0:
                XMGNP = float(MGNTPP[N]) / float(LOLTPP[N])
            else:
                XMGNP = 0.0

            if INDX != 1:
                f.write(
                    "      %2d     %4d     %8.2f  %8d     %4d     %8.2f            %s"
                    % (N, LOLTHP[N], XMGNH, MGNTHP[N], LOLTPP[N], XMGNP, XT)
                )
            else:
                f.write(
                    "      %2d         %8.2f  %8.2f            %s"
                    % (N, LOLTPP[N], XMGNP, XT)
                )

            if LOLSHP[N] > 0:
                XMGNH = float(MGNSHP[N]) / float(LOLSHP[N])
            else:
                XMGNH = 0.0

            if LOLSPP[N] > 0:
                XMGNP = float(MGNSPP[N]) / float(LOLSPP[N])
            else:
                XMGNP = 0.0

            if INDX != 1:
                f.write(
                    "      %2d     %4d     %8.2f  %8d     %4d     %8.2f            %s"
                    % (N, LOLSHP[N], XMGNH, MGNSHP[N], LOLSPP[N], XMGNP, XS)
                )
            else:
                f.write(
                    "      %2d         %8.2f  %8.2f            %s"
                    % (N, LOLSPP[N], XMGNP, XS)
                )

        if WOLGHP > 0.0:
            XMGNH = WGNGHP / WOLGHP
        else:
            XMGNH = 0.0

        if WOLGPP > 0.0:
            XMGNP = WGNGPP / WOLGPP
        else:
            XMGNP = 0.0

        if INDX != 1:
            f.write(
                "      %s     %7.2f  %8.2f   %7.0f     %7.2f  %8.2f            %s"
                % (XA, WOLGHP, XMGNH, WGNGHP, WOLGPP, XMGNP, XG)
            )
        else:
            f.write(
                "      %s         %8.2f  %8.2f            %s" % (XA, WOLGPP, XMGNP, XG)
            )

        if WOLTHP > 0.0:
            XMGNH = WGNTHP / WOLTHP
        else:
            XMGNH = 0.0

        if WOLTPP > 0.0:
            XMGNP = WGNTPP / WOLTPP
        else:
            XMGNP = 0.0

        if INDX != 1:
            f.write(
                "      %s     %7.2f  %8.2f   %7.0f     %7.2f  %8.2f            %s"
                % (XA, WOLTHP, XMGNH, WGNTHP, WOLTPP, XMGNP, XT)
            )
        else:
            f.write(
                "      %s         %8.2f  %8.2f            %s" % (XA, WOLTPP, XMGNP, XT)
            )

        if WOLSHP > 0.0:
            XMGNH = WGNSHP / WOLSHP
        else:
            XMGNH = 0.0

        if WOLSPP > 0.0:
            XMGNP = WGNSPP / WOLSPP
        else:
            XMGNP = 0.0

        if INDX != 1:
            f.write(
                "      %s     %7.2f  %8.2f   %7.0f     %7.2f  %8.2f            %s"
                % (XA, WOLSHP, XMGNH, WGNSHP, WOLSPP, XMGNP, XS)
            )
        else:
            f.write(
                "      %s         %8.2f  %8.2f            %s" % (XA, WOLSPP, XMGNP, XS)
            )

    f.close()  # close the file "output"

    # Zero out statistics reated arrays and scalars:
    for IAR in range(NOAREA):
        WGNGHA[IAR] = 0.0
        WGNTHA[IAR] = 0.0
        WGNSHA[IAR] = 0.0
        WOLGHA[IAR] = 0.0
        WOLTHA[IAR] = 0.0
        WOLSHA[IAR] = 0.0

    for IAR in range(NOAREA):
        WGNGPA[IAR] = 0.0
        WGNTPA[IAR] = 0.0
        WGNSPA[IAR] = 0.0
        WOLGPA[IAR] = 0.0
        WOLTPA[IAR] = 0.0
        WOLSPA[IAR] = 0.0

    for IAR in range(NOAREA):
        WGNSPA[IAR] = 0.0
        WOLSPA[IAR] = 0.0

    WGNSHP = 0.0
    WOLSHP = 0.0
    WGNSPP = 0.0
    WOLSPP = 0.0
    WGNGHP = 0.0
    WOLGHP = 0.0
    WGNGPP = 0.0
    WOLGPP = 0.0
    WGNTHP = 0.0
    WOLTHP = 0.0
    WGNTPP = 0.0
    WOLTPP = 0.0

    for N in range(NFCST):
        for IAR in range(NOAREA):
            LOLTHA[IAR, N] = 0
            LOLGHA[IAR, N] = 0
            MGNTHA[IAR, N] = 0
            MGNGHA[IAR, N] = 0
            LOLTPA[IAR, N] = 0
            LOLGPA[IAR, N] = 0
            MGNTPA[IAR, N] = 0
            MGNGPA[IAR, N] = 0
        LOLTHP[N] = 0
        LOLGHP[N] = 0
        MGNTHP[N] = 0
        MGNGHP[N] = 0
        LOLTPP[N] = 0
        LOLGPP[N] = 0
        MGNTPP[N] = 0
        MGNGPP[N] = 0

    if IYEAR <= 5:  # maybe use < 5; check back later
        f.write("  KVs = %4d %4d %4d %4d %4d" % (KWHERE, KVWHEN, KVSTAT, KVTYPE, KVLOC))

    # Begin checking for convergence
    if KWHERE == 1:
        if KVWHEN == 1:
            if KVSTAT == 1:
                # LOOP FOR HOURLY, AREA, LOLE
                if KVTYPE == 1:
                    SUM = SWLSHA[KVLOC]
                else:
                    SUM = SOLSHA[KVLOC, NOERR]
            # KVSTAT = 2, LOOP FOR HOURLY, AREA, EUE
            else:
                if KVTYPE == 1:
                    SUM = SWNSHA[KVLOC]
                else:
                    SUM = SGNSHA[KVLOC, NOERR]
        # KVWHEN = 2, LOOP FOR PEAK, AREA, LOLE
        else:
            if KVSTAT == 1:
                if KVTYPE == 1:
                    SUM = SWLSPA[KVLOC]
                else:
                    SUM = SOLSPA[KVLOC, NOERR]
            # KVWHEN=2, KVSTAT = 2, LOOP FOR PEAK, AREA, EUES
            else:
                if KVTYPE == 1:
                    SUM = SWNSPA[KVLOC]
                else:
                    SUM = SGNSPA[KVLOC, NOERR]
    else:
        if KVWHEN == 1:
            if KVSTAT == 1:
                # LOOP FOR HOURLY, POOL, LOLE
                if KVTYPE == 1:
                    SUM = SWLSHP
                else:
                    SUM = SOLSHP[NOERR]
            # KVSTAT = 2, LOOP FOR HOURLY, POOL, EUE
            else:
                if KVTYPE == 1:
                    SUM = SWNSHP
                else:
                    SUM = SGNSHP[NOERR]
        # KVWHEN = 2, LOOP FOR PEAK, POOL, LOLE
        else:
            if KVSTAT == 1:
                if KVTYPE == 1:
                    SUM = SWLSPP
                else:
                    SUM = SOLSPP[NOERR]
            # KVWHEN=2, KVSTAT = 2, LOOP FOR PEAK, POOL, EUES
            else:
                if KVTYPE == 1:
                    SUM = SWNSPP
                else:
                    SUM = SGNSPP[NOERR]

    RFLAG, SSQ, XLAST = cvchk(CLOCK, FINISH, IYEAR, CVTEST, SUM, XLAST, SSQ)
    NOERR = 1

    if IYEAR == INTVT:
        intm(
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
        )

        INTVT += INTV

    else:  # if IYEAR != INTVT
        assert type(IYEAR) is int
        IFIN = int(FINISH / 8760)
        if IYEAR == IFIN:
            intm(
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
            )

    if RFLAG == 1 or CLOCK >= FINISH:
        ITAB, SUSTAT = report(
            IYEAR,
            ITAB,
            INDX,
            SUSTAT,
            DPLOLE,
            EUES,
            HLOLE,
            LSTEP,
            NAMA,
            NFCST,
            SGNGHA,
            SGNGHP,
            SGNGPA,
            SGNGPP,
            SGNSHA,
            SGNSHP,
            SGNSPA,
            SGNSPP,
            SGNTHA,
            SGNTHP,
            SGNTPA,
            SGNTPP,
            SOLGHA,
            SOLGHP,
            SOLGPA,
            SOLGPP,
            SOLSHA,
            SOLSHP,
            SOLSPA,
            SOLSPP,
            SOLTHA,
            SOLTHP,
            SOLTPA,
            SOLTPP,
            SWLGHA,
            SWLGHP,
            SWLGPA,
            SWLGPP,
            SWLSHA,
            SWLSHP,
            SWLSPA,
            SWLSPP,
            SWLTHA,
            SWLTHP,
            SWLTPA,
            SWLTPP,
            SWNGHA,
            SWNGHP,
            SWNGPA,
            SWNGPP,
            SWNSHA,
            SWNSHP,
            SWNSPA,
            SWNSPP,
            SWNTHA,
            SWNTHP,
            SWNTPA,
            SWNTPP,
            XNEWA,
            XNEWP,
            XG,
            XT,
            XS,
            XA,
            FH,
            SH,
        )

    return ATRIB, IPOINT, MFA, NUMINQ, SSQ, XLAST, RFLAG, INTVT, ITAB, SUSTAT
