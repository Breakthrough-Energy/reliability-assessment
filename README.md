# reliability-assessment
Reliability Assessment Module

**This readme file will get continuous updating based on the development progress. The description of the method logic and the input/output data formats are subjected to change based on later code-optimization during the development.**

The goal of this project is to port an original Fortran-based power system reliability assessment program ('NARP') to Python. The program will calculate several useful reliability indices, including LOLE (Loss of Load Expectations),  Hourly LOLE (HLOLE in short), EUE (Expected Unserved Energy, in kWh or MWh) for transmission power systems. Those reliability indices measure the availability of generation and transmission capacity given a preset table of failure probabilities.

## General Program Logic
The simulation proceeds sequentially in chronological order. As the target is to compute hourly relibility indices, the temporal granularity of the simulation is hourly. Daily peak reliability indices are calculated in a similar way except the process only steps through daily peaks, which is defined as the time of interconnected system peak. 

1) In each hour, the status of each generator and transmission link is randomly and independently drawn according to the probability distribution of generating unit and transmission link states specified by the user. _(As an alternate, generator and transmission link states can be drawn once per day. It is recommended that generator and transmission link states be drawn once an hour if hourly reliability indices (HLOLE and EUE) are of significant or primary interest. However, if daily peak indices are of sole or primary interest, it is recommended that generator and transmission link states are drawn once per day. The effect of this is reduced computational time at the expense of less rapid convergence of the hourly reliability indices.)_

2) The generating capacity available in each area is determined by the summation of available capacities of individual generating units. Two capacities are found for each area:
    1. the capacity associated with generating units located in the area regardless of ownership, and
    2. the capacity associated with generating units owned by the area regardless of location.

3) The native load for each area is updated to the current hour.

4) Scheduled transfers between areas are determined from input data. 
	1. Similarly, transfers associated with jointly-owned or out-of-area generating units are determined based on the availability statuses of the units and the area ownership percentages. 
	2. Scheduled transfers and transfers associated with jointly-owned units are added algebraically to give net scheduled transfers between areas.

5) The margin in each area is found by subtracting area native load from available generating capacity located in the area.
If this margin is positive for all areas, the clock advances to the next hour, and the process is repeated. _(If load forecast uncertainty is being modeled, the above process is begun with the highest load scenario in all areas. If all area margins are positive, no lower load scenarios need to be considered, and the clock is advanced to the next hour.)_

6) If any area has a negative margin, the following procedure is followed to obtain the necessary relief from other areas.

	1. If only one area has a negative margin, a simplified test not requiring a load flow solution is tried first. 
		1. In this approach, the total capacity assistance available from areas directly connected to the negative­ margin area is found. Here the capacity assistance available from an area is the minimum of: (i) the area margin or (ii) the capacity of the transmission link between the area and the negative-margin area. 
		2. The total available capacity assistance from areas directly connected to the negative-margin area is just the sum of the capacity assistances available from these areas. 
		3. Then, if the total available capacity assistance is greater than the capacity shortfall in the negative­ margin area, it is assumed that adequate capacity can be imported to eliminate the load loss, and the clock is advanced to the next hour. Otherwise, the transmission module is called. 

	2. If more than one area has a negative margin, the transmission module is called directly.

7) The DC load flow transmission module employs a two-step procedure to eliminate loss-of-load events. 

	1. First, a load flow is conducted using area loss injections associated with net scheduled transfers as well as injections associated with desired emergency transfers. If this load flow can be solved without violation of transmission capacity limits, the clock is advanced to the next hour. 

	2. Otherwise, a linear programming approach is employed to find a feasible load flow solution or minimize the amount of load loss. _(The linear programming approach can consider two modes of emergency assistance: loss-sharing or no-loss sharing. Further, the linear programming approach can enforce constraints on the sum of flows around each area.)_ **If load loss persists after the optimization step, load-loss statistics are collected.**

8) The simulation process is continued until the specified number of years have been simulated or the specified convergence criterion has been satisfied.

## Method
This program is area-based, such that;
1) No realistic transmission line is modeled like normal power flow but "equivalent transmission links" between each area pair.
2) Internal transmission lines within each area are ignored.
3) Monte Carlo simulation method is utilized to calculate the statistics for the above-mentioned reliability indices. One note for the audience without power system reliably background: in the power system reliability area, the phrase 'Monte Carlo simulation' does NOT represent a concrete 'algorithm', BUT, a *higher-level abstract research idea* (by analog, 'quicksort' is a specific 'algorithm'; but its *background idea* is 'divide-and-conquer'). Thus, there are NO specific modules or functions in this repository *explicitly or deliberately* with the naming 'Monte Carlo.' That is the standard practice in the power system reliability area.

## Input 

The raw data input files are three files (note: no extension name): 

INPUTB -- the base case file (required)

INPUTC -- the "shadow" file of INPUTB, i.e., used to modify certain data sections in the INPUTB file. (OPTIONAL)

LEEI -- the one-year hourly load file for each area 

## Output

The output of the simulation will contain:

1) System descriptions (rephrased and re-arranged information extracted from INPUTB file), e.g., how many areas are contained.
2) Chosen method options, e.g., Specifications for convergence test of the Monte Carlo simulation.
3) Summary of numerical results: summary statistics (LOLE, HLOLE, EUE) for the pool (all areas); (discrete) probability distribution table (LOLE, HLOLE, EUE) for each area; other meta-information during calculation or Monte Carlo simulation.

## Flowchart

(up to refine or change)

<img width="534" alt="new_flowchart" src="https://user-images.githubusercontent.com/45750983/197419657-156b97bf-eb3d-4d47-97ff-3807850d5743.png">

## Features (last updated on Sep.27.2021)
|  | NARP (Ours) | PRAS (NREL) | Impact on Reliability Indices |
| ------------- | ------------- | ------------- | ------------- |
| Transmission model | Consider admittance | Not consider admittance <br /> (only capacity based) | More optimistic (PRAS) |
| Generator model | Offer both 3- and 2-state | 2-state only | Less realistic for larger units (PRAS) |
| Load loss sharing mode | Can consider | N/A | This can help decide which mode <br /> of operation is more beneficial to <br /> reliability (NARP) |
| Restart capability | Yes | N/A | This can help save time when higher <br /> accuracy is desired by continuing <br /> from last snaptshot (NARP) |
| Methodologies | Sequential Monte Carlo <br /> (equal time interval) | Convolution method (analytic) <br /> Sequential Monte Carlo <br /> Non-sequential Monte Carlo | The Convolution method is not <br /> suitable for large systems | 
| Energy Storage | No | Yes | NARP has the potential to be <br /> adapted for this purpose |


## License

The source code for the site is licensed under the MIT license, which you can find in
the MIT-LICENSE.txt file.
