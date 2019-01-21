# Randomized Optimization

This assignment uses the ABAGAIL java library provided by the GT faculty and edited code provided by Jonathan Tay's GitHub.


### Files:
1. [Analysis.pdf](Analysis.pdf)
    * Randomized Optimization Analysis
2. ABAGAIL
    * Collection of Java packages that implement machine learning and AI algorithms
3. datasets/
    * Data split into 60% train, 20% validation, 20% test
4. Part 1/ (contains part 1 of the assignment)
    1. NN0.py
        * Code for Back-propagation training of neural network
    2. NN1.py
        * Code for Randomized Hill Climbing training of neural network
    3. NN2.py
        * Code for Simulated Annealing training of neural network
    4. NN3.py
        * Code for Genetic Algorithm training of neural network
    5. NN_OUTPUT/ 
        * BP, GA, RHC, and SA folders contain output from the code
    6. Charts/
        * Charts folder contains output in an excel workbook used to produce graphs
5. Part 2/ (contains part 2 of the assignment)
    1. continuouspeaks.py
        * Code to solve the Continuous Peaks problem
    2. flipflop.py
        * Code to solve the Flip Flop problem
    3. tsp.py
        * Code to solve the Traveling Salesman Problem
    4. Charts/
        * Jupyter Notebooks for hyperparameter tuning of the 4 algorithms for the CP, FF, and TSP problems
        * Part2CombinedAnalysis.xlsx - charts and combined analysis
    5. CONTPEAKS/
        * Output from continuouspeaks.py
    6. FLIPFLOP/
        * Output from flipflop.py
    7. TSP/
        * Output from tsp.py

---
**Languages:**  
Python 3.6, Jython, Java

**Libraries:**  
ABAGAIL, Pandas, NumPy, Matplotlib

**Environments:**  
Jupyter Notebooks

**Software:**  
Microsoft Excel

---
## Instructions to Run:
1. cd into Part 1 folder. In terminal run:
    ```bash
    jython NN0.py
    jython NN1.py
    jython NN2.py
    jython NN3.py
    ```
2. Output from code is manually combined into excel workbooks in the Charts folder and graphed.
2. cd into Part 2 folder. In terminal run:
    ```bash
    jython continuouspeaks.py
    jython flipflop.py
    jython tsp.py
    ```
3. cd into Charts folder and run each Jupyter Notebook in listed order. For each of the three toy problems CP, FF, and TSP, run in order of RHC, SA, GA, and MIMIC. This will produce output used for hyperparameter selection. 
4. Results are manually combined into Part2CombinedAnalysis.xlsx and charts are plotted in Excel.
