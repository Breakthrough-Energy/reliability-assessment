# reliability-assessment
Reliability Assessment Module

The goal of this project is to port an original Fortran-based power system reliability assessment program ('NARP') to Python. The program will calculate several useful reliability indices, including LOLE (Loss of Load Expectations),  Hourly LOLE (HLOLE in short), and EUE (Expected Unserved Energy, in kWh or MWh) for transmission power systems. Those reliability indices measure the availability of generation and transmission capacity given a preset table of failure probabilities.

## Method

Similar to the [*GE-MARS*](https://www.geenergyconsulting.com/practice-area/software-products/mars) or other industry simulation applications, this program is area-based, which is designed for zonal reliability analysis. Nodal models with full granularity need to be equivalented into a zonal model first in order to be able to use this program. More specifically:
1) Equivalent transmission links (i.e., tie-line) between each area pair are considered and modeled.
2) Internal transmission lines within each area are ignored.
3) Monte Carlo simulation method is utilized to calculate the statistics for the above-mentioned reliability indices. 


## General Program Logic
The simulation proceeds sequentially in chronological order. As the target is to compute hourly reliability indices, the temporal granularity of the simulation is hourly. Daily peak reliability indices are calculated similarly, except the process only steps through daily peaks, defined as the time of interconnected system peaks. 

1) In each hour, the status of each generator and transmission link is randomly and independently drawn according to the probability distribution of generating unit and transmission link state specified by the user. _(As an alternate, generator and transmission link states can be drawn once per day. It is recommended that generator and transmission link states be drawn once an hour if hourly reliability indices (HLOLE and EUE) are of significant or primary interest. However, if daily peak indices are of sole or primary interest, it is recommended that generator and transmission link states are drawn once per day. This effect reduces computational time at the expense of less rapid convergence of the hourly reliability indices.)_

2) The generation capacity available in each area is determined by the summation of the available capacities of individual generator units. Two capacities are found for each area:
    1. the capacity associated with generating units located in the area regardless of ownership, and
    2. the capacity associated with generating units owned by the area regardless of location.

3) The native load for each area is updated to the current hour.

4) Scheduled transfers between areas are determined from input data. 
	1. Similarly, transfers associated with jointly-owned or out-of-area generating units are determined based on the availability statuses of the units and the area ownership percentages. 
	2. Scheduled transfers and transfers associated with jointly-owned units are added algebraically to give net scheduled transfers between areas.

5) The margin in each area is found by subtracting the area's native load from the available generating capacity located in the area.
If this margin is positive for all areas, the clock advances to the next hour, and the process is repeated. _(If load forecast uncertainty is being modeled, the above process is begun with the highest load scenario in all areas. If all area margins are positive, no lower load scenarios need to be considered, and the clock is advanced to the next hour.)_

6) If any area has a negative margin, the following procedure is followed to obtain the necessary relief from other areas.

	1. If only one area has a negative margin, a simplified test not requiring a load flow solution is tried first. 
		1. In this approach, the total capacity assistance available from other areas directly connected to the negative­margin area is found. Here the capacity assistance available from an area is the minimum of (i) the area margin or (ii) the capacity of the transmission link between the area and the negative-margin area. 
		2. The total available capacity assistance from areas directly connected to the negative-margin area is just the sum of the capacity assistances available from these areas. 
		3. Then, if the total available capacity assistance is greater than the capacity shortfall in the negative­ margin area, it is assumed that adequate capacity can be imported to eliminate the load loss, and the clock is advanced to the next hour. Otherwise, the transmission module is called. 

	2. If more than one area has a negative margin, the transmission module is called directly.

7) The DC load flow transmission module employs a two-step procedure to eliminate loss-of-load events. 

	1. First, a load flow is conducted using area loss injections associated with net scheduled transfers and injections associated with desired emergency transfers. The clock is advanced to the next hour if this load flow can be solved without violation of transmission capacity limits. 

	2. Otherwise, a linear programming approach is employed to find a feasible load flow solution or minimize the amount of load loss. _(The linear programming approach can consider two modes of emergency assistance: loss-sharing or no-loss sharing. Further, the linear programming approach can enforce constraints on the sum of flows around each area.)_ **If load loss persists after the optimization step, load-loss statistics are collected.**

8) The simulation process is continued until the specified maximum number of years has been simulated or the specified convergence criterion has been satisfied.


## Installation

1) This module requires Python v3.8 with NumPy v1.20 and pandas as the prerequisites. 
2) If [Anaconda](https://www.anaconda.com/products/distribution) (recommended) is used as the Python package manager, it will install all the requirements automatically. 
3) Git clone or download the zipped file of the package from [our repo](https://github.com/Breakthrough-Energy/reliability-assessment/tree/main/reliabilityassessment/). Put its content to a location per user’s preference, e.g., `C:/reliability assessment/`. 
4) In Anaconda prompt or other proper cmd prompts, navigate to the folder of the subfolder of the code (“reliability-assessment”; do not forget the "-" symbol in between), and run `pip install .` (do not forget the dot symbol in the end). This will install the whole package as a third-party library in your working environment. 

## Running the Program 

### Input
1) Before running the program, the user must prepare several raw input files that define the system for reliability assessment and hyperparameters for simulation settings. 
2) These inputs are described in csv file format (except for the LEEI file, which defines the load profile data that follows the classic EEI format used in many industry applications like [*PROMOD*](https://www.hitachienergy.com/us/en/products-and-solutions/energy-portfolio-management/enterprise/promod)), as described in the following table.
3) An example set of simulation-ready input files is available in our repo.

| Input | Format | Description |
| ------------ | ------------- | ------------- | 
| ZZTC |  csv | Simulation title card. 
| ZZMC |  csv | Miscellaneous program settings, including: Seed (Loss sharing included); Definition of seasons; Convergence test parameters; Frequency of drawing of the status of generators and transmission lines, either daily or hourly; Frequency of data collection, either daily peak or hourly; Specifications for the probability distribution of EUE (e.g.upper limit); Printout options. 
| ZZLD |  csv | System data, including: Area name; Peak demand; Forecast uncertainty; Outage window; Forbidden period; Flow constraints. 
| ZZUD |  csv | Unit data, including: Unit Name; Serial number; Location; Capacity; Outage rate; Derated outage rate; Percentage derating due to partial failure; Option to predetermine or auto-schedule; Beginning week and Duration of the first and second weeks. 
| ZZFC |  csv | Firm contracts, specifying the firm interchanges of power between areas. 
| ZZOD |  csv | Unit ownership data in percentage. 
| ZZTD |  csv | Line data, including: Location of areas; Admittance (negative number); Capacity of the line in the forward direction; Capacity of the line in the reverse direction; Probability of the line in each state (six states in total for each line). 
| ZZDD |  csv | Line derating data. 
| LEEI |  EEI | Load profile data. 

4) The program provides automatic error checking for data anomalies to improve the robustness of user-defined inputs given the simulation needs.
5) Please note that among all the input files, ZZMC, ZZUD, ZZLD, ZZTD, ZZTC, and LEEI are **required** to run a simulation, while other files are optional. 

After preparing all the required inputs, navigate to the folder where one puts the entire inputs related files, specify the path and  run:  
```python
INPUT_DIR = "full path of the input data”
narpMain(INPUT_DIR)
```
 
The command triggers the input data processing function “dataf1()” followed by the Monte-Carlo function “contrl()”. Data processing is a one-time processing while Monte-Carlo is a repeated iteration; thus, convergence needs to be watched. While the program is progressing, it will report the iteration convergence index information as below in the command window: 
```python
... 
n-th REPLICATION; SUM=143.0 NEW=0.0 MEAN=0.43202416918429004 CONVERGENCE=3.702682434420153 
... 
```
Note that: 
- When the “CONVERGENCE” drops to below 1, it indicates that the program has reached convergence tolerance and finished evaluation. 
- If it cannot reduce to less than 1 within predefined maximum iterations, it indicates that either the given power system itself has physics issues or the simulation data needs to be checked. The user can follow the following steps to investigate potential problems and identify solutions: 

1. Increase the predefined number of maximum iterations (default value is 9999) and re-run the whole program from the beginning (or continue from a saved snapshot file); 
2. If it still diverges even by using a reasonably larger "maximum # of iterations", carefully check the raw input data and look for any possible anomalies; 
3. If the raw input data is correct after manual inspection, then the divergence of the Monte Carlo simulation indicates that: 
	- Either the power system itself has significant reliability problems that block the program from converging, or
	- There may be (implicit) numeric issues during the simulation of the given system, which may require further debugging.


### Output

After the simulation finishes successfully with convergence,  an "output" text file will be generated in the project working directory. In the output file, reliability statistics for all the areas are printed out in a table format. The final output file contains the following:

1) System descriptions (rephrased and re-arranged information extracted from INPUTB file), e.g., how many areas are contained.
2) Chosen method options, e.g., Specifications for convergence test of the Monte Carlo simulation.
3) Summary of numerical results: summary statistics (LOLE, HLOLE, EUE) for the pool (all areas); (discrete) probability distribution table (LOLE, HLOLE, EUE) for each area; other meta-information during calculation or Monte Carlo simulation.

## Flowchart

(up to refine or change)

![new_flowchart](https://user-images.githubusercontent.com/45750983/201003713-9491169c-7811-4058-b9aa-2faa1010fc3f.png)

## Features (last updated on Sep.27.2021)
|  | NARP (Ours) | PRAS (NREL) | Impact on Reliability Indices |
| ------------- | ------------- | ------------- | ------------- |
| Transmission model | Consider admittance | Not consider admittance <br /> (only capacity based) | More optimistic (PRAS) |
| Generator model | Offer both 3- and 2-state | 2-state only | Less realistic for larger units (PRAS) |
| Load loss sharing mode | Can consider | N/A | This can help decide which mode <br /> of operation is more beneficial to <br /> reliability (NARP) |
| Restart capability | Yes | N/A | This can help save time when higher <br /> accuracy is desired by continuing <br /> from last snaptshot (NARP) |
| Methodologies | Event triggered Sequential Monte Carlo <br /> (equal time interval) | Convolution method (analytic) <br /> Sequential Monte Carlo <br /> Non-sequential Monte Carlo | The Convolution method is not <br /> suitable for large systems | 
| Energy Storage | No | Yes | NARP has the potential to be <br /> adapted for this purpose |


## License

The source code for the site is licensed under the MIT license, which you can find in
the MIT-LICENSE.txt file.
