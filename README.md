# reliability-assessment
Reliability Assessment Module

**This readme file will get continuous updating based on the development progress. The description of the method logic and the input/output data formats are subjected to change based on later code-optimization during the development.**

The program will calculate several useful reliability indices, including LOLE (Loss of Load Expectations),  Hourly LOLE (HLOLE in short), EUE (Expected Unserved Energy, in kWh or MWh) for transmission power systems. Those reliability indices measure the availability of generation and transmission capacity given a preset table of failure probabilities.  

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

## License

The source code for the site is licensed under the MIT license, which you can find in
the MIT-LICENSE.txt file.
