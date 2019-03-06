# Supervised Learning

**Objective:**  
Analyze five different supervised learning algorithms – Decision Trees, Boosting, Neural Networks, Support Vector Machines, and k-Nearest Neighbors on two different datasets.

**Datasets:**  
1. [Faulty Steel Plates](http://archive.ics.uci.edu/ml/datasets/steel+plates+faults) (1,941 instances, 27 attributes, 7 labels)
2. [Phishing Websites](https://archive.ics.uci.edu/ml/datasets/phishing+websites) (11,055 instances, 30 attributes, binary label)

WEKA GUI 3.8.2, a suite of machine learning software written in Java, is used to generate the output for the analysis. The step-by-step instructions to produce the output is listed in WekaInstructions.pdf. Training, validation, and test accuracy from WEKA is inputted in Excel and learning curves are generated in Excel. See Analysis.pdf for analysis of output/results.

### Files:
1. [Analysis.pdf](Analysis.pdf)
    * Supervised Learning Analysis
2. datasets/
   1. train_faults.arff
   2. test_faults.arff
   3. train_phishing.arff
   4. test_phishing.arff
3. WekaInstructions.pdf
    * Weka instructions for generating output
4. LearningCurves/WekaOutput.xlxs
    * Output from Weka plotted in Excel charts

### Software:
1. WEKA GUI 3.8.2
2. Microsoft Excel

### Data:
Separated 80%-20% by WEKA GUI as train/validation and test set
