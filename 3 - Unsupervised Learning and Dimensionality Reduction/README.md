# Unsupervised Learning and Dimensionality Reduction

**Objective:**  
Six different algorithms are implemented; the first two are clustering – k-means clustering and Expectation Maximization and the last four are dimensionality reduction algorithms – PCA, ICA, Randomized Projections, and Random Forest. The experiments are split into four main parts and used to train a Neural Network. All algorithms are evaluated using Scikit-learn in Python.

**Datasets:**  
1. [Faulty Steel Plates](http://archive.ics.uci.edu/ml/datasets/steel+plates+faults) (1,941 instances, 27 attributes, 7 labels)
2. [Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) (569 instances, 31 attributes, binary label)

### Files:
1. [Analysis.pdf](Analysis.pdf)
    * Unsupervised Learning and Dimensionality Reduction Analysis
2. Datasets/
    * Data split into 80% train, 20% test
3. Output/ (Output from code)
    1. BASE/
        * Clustering for both datasets and clustering then neural network output
    2. PCA/
        * Principal Component Analysis clustering output for both datasets
    3. ICA/
        * Independent Component Anlaysis clustering output charts for both datasets
    4. RP/
        * Random Projection clustering output for both datasets
    5. RF/
        * Random Projection clustering output for both datasets
4. Charts/ 
    1. BASE.ipynb
        * Clustering analysis and clustering then neural network charts for both datasets
    2. PCA.ipynb
        * Principal Component Analysis and clustering analysis charts for both datasets
    3. ICA.ipynb
        * Independent Component Anlaysis and clustering analysis charts for both datasets
    4. RP.ipynb
        * Random Projection and clustering analysis charts for both datasets
    5. RF.ipynb
        * Random Forest and clustering analysis charts for both datasets
    6. Dimensional Reduction and Neural Network.ipynb
        * Dimensional Reduction and Neural Network comparisons between BASE, PCA, ICA, RP, and RF
5. Code
    1. parse.py
        * Code for converting training and test data csv files into hdf output
    2. benchmark.py
        * Code for benchmark output for the Neural Network
    3. clustering.py
        * Code for clustering output for BASE, PCA, ICA, RP, and RF
    4. helpers.py
        * Formulas for evaluating clusters
    5. PCA.py
        * Code for Principal Component Analaysis
    6. ICA.py
        * Code for Independent Component Analaysis
    7. RP.py
        * Code for Random Projection Analysis
    8. RF.py
        * Code for Random Forest Analysis
    9. cluster.sh
        * Runs clustering.py for BASE, PCA, ICA, RP, and RF
    10. dimred.sh
        * Runs PCA.py, ICA.py, RP.py, and RF.py
        
---
**Languages:**  
Python 3.6

**Libraries:**  
Pandas, NumPy, Scikit-learn, SciPy, Matplotlib

**Environments:**  
Jupyter Notebooks

---
## Instructions to Run:
1. Run code to convert dataset csv files into hdf output. In terminal, run:  

    ```bash
    python parse.py
    ```
2. Run code to get the benchmark for the Neural Network. In terminal, run:  

    ```bash
    python benchmark.py
    ```
3. Run code for clustering analysis for BASE used in Part 1. In terminal, run:  

    ```bash
    python clustering.py BASE
    ```
4. Run code for dimensionality reduction algorithms for Parts 1, 2, 3, and 4. In terminal, run:  

    ```bash
    ./dimred.sh
    ```
5. In PCA.py, ICA.py, RP.py, and RF.py, based on output from Data for 2 - "dim red.csv", find the best number of dimensions and update dims variable under Data for 3. Re-run code, in terminal:  

    ```bash
    ./dimred.sh
    ```
6. Run code for clustering after dimensionality reduction. Used for Parts 2 and 3. In terminal, run:  

    ```bash
    ./cluster.sh
    ```
7. Run all Jupyter Notebooks to produce plots and charts using data from the /Output folder
