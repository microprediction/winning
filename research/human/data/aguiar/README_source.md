README Document for Reproducing Results in
==========================================
"Random Utility and Limited Consideration"
=============================================
Victor H. Aguiar
vaguiar@uwo.ca

Maria Jose Boccardi
majoboccardi@gmail.com

Nail Kashaev
nkashaev@uwo.ca

Jeongbin Kim
jbkimecon@gmail.com

Software
========

Version 1.6.4 of the `Julia` programming language was used in coding the analysis files. For details about how to install `Julia` on different platforms and how to make `Julia` programs executable from the command line see <https://julialang.org/downloads/platform/>. After installation of `Julia 1.6.4.` run `using Pkg` and `Pkg.instantiate()` in the `Julia` terminal after setting the replication folder as the main one.

Some simulations and estimations use `KNITRO 12.3`.  

Hardware
========

- The code was run on Mac mini (M1, 2020) with 16 Gb of RAM
- Expected Execution Time (approximately): main text -- 40 min; Appendix E -- 140 hours; Appendix F -- 7 min.


Content
=======

-   `appendix_E`  -- the folder contains the analysis files to replicate the results in Appendix E.

-   `appendix_F`  -- the folder contains the analysis files to replicate the results in Appendix F.

-   `data`  -- the folder contains the data used in the application (Sections 4.2 and 5.2).

-   `main`  -- the folder contains the analysis files to replicate the results in Section 5.2.

-   `tables and figures`  -- the folder contains the analysis files to replicate all tables and figures in the paper.

-   `Manifest.toml` and `Project.toml`  -- toml files with all necessary `Julia` packages.



Below, we describe the content of every folder.

`appendix_E`
============

-   `power_results` --  `csv` files contain the results of power simulations.

-   `functions_common_power.jl` -- the functions used in `testing_power.jl`.

-   `test_power.sh` --  this script runs `testing_power.jl` with different input parameters.

-   `testing_power.jl` -- the code generates the results in Appendix E.



`appendix_F`
============

-   `results` -- the `csv` files with p-values used in Appendix F.

-   `functions_common_testing.jl` -- the functions used in `testing_LA_hom.jl`.

-   `test_LA_hom.sh` --  this script runs `testing_LA_hom.jl`.

-   `testing_LA_hom.jl` -- the code generates the results in Appendix F.




`data`
============

-    `csv` files that contain the data for high, low, and medium cost frames used in Sections 4.2 and 5.2.


`main`
============

-   `results` -- the `csv` files that contain the value of the test statistic and its p-value used in Table 2 for LA,EBA, and RUM models.

-   `functions_common_testing.jl` -- the functions used in `testing_stable.jl`.

-   `test_stable.sh` --  this script runs `testing_stable.jl` with different input parameters.

-   `testing_stable.jl` -- the code generates the results in Section 5.2.


`tables and figures`
============

-   `results` -- the `csv` and `pdf` files with tables and figures.

-   `tabandfig.jl`-- the code generates all tables and figures.


`choice alternative identifier` in csv files in `data`
- Prizes: Z=(50,48,30,14,12,10,0) tokens
- Lottery 0:  l0=(0,0,0,0,1,0,0)
- Lottery 1: l1 = (1/2, 0, 0, 0, 0, 0, 1/2)
- Lottery 2: l2 = (0, 0, 1/2, 0, 0, 1/2, 0)
- Lottery a = (0, 2/5, 0, 3/10, 0, 0, 3/10) (not in the choice set)
- Lottery 3:  l3 = 1/2*l1 +1/2*l2 
- Lottery 4: l4 = 1/2*l1+1/2*a
- Lottery 5:  l5 = 1/2*l2 + 1/2*a

`attention cost/frame`  
- high cost: 5 operations
- medium cost: 3 operations
- low cost: 1 operation