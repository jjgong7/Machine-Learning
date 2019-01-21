# Markov Decision Processes

### Files:
1. [Analysis.pdf](Analysis.pdf)
    * Markov Decision Processes Analysis
2. BURLAP/
    * The Brown-UMBC Reinforcement Learning and Planning (BURLAP) java code library is for the use and development of single or multi-agent planning and learning algorithms and domains to accompany them
3. gridworld.html
    * Used for creating the grid world matrix
4. Solution/
    1. easyGW.py
        * Code for Easy Grid World
    2. hardGW.py
        * Code for Hard Grid World
    3. Easy Grid World Output. Includes .csv output for:
        1. Easy Policy Iteration
        2. Easy Value Iteration
        3. Easy Q-Learning
    4. Hard Grid World Output. Includes .csv output for:
        1. Hard Policy Iteration
        2. Hard Value Iteration
        3. Hard Q-Learning
5. Graphs/
    1. Q-Learning Easy.ipynb
        * Comparison charts for Q-learning hyperparameter selection
    2. Q-Learning Hard.ipynb
        * Comparison charts for Q-learning hyperparameter selection
    3. Easy Comparison - Policy, Value, Q-Learning.ipynb
        * Comparison charts for the three different algorithms
    4. Hard Comparison - Policy, Value, Q-Learning.ipynb
        * Comparison charts for the three different algorithms

---
**Languages:**  
Python 3.6, Jython, Java

**Libraries:**  
BURLAP, Pandas, NumPy, Matplotlib

**Environments:**  
Jupyter Notebooks

---
## Instructions to Run:
1. cd into Solution folder. Run code for Easy Grid World and Hard Grid World. In terminal, run:  
 
    ```bash
    jython -J-Xms6000m -J-Xmx6000m easyGW.py
    jython -J-Xms6000m -J-Xmx6000m hardGW.py
    ```
2. Policy Iteration, Value Iteration, and Q-Learning charts are produced as output from the code. 
3. Run the following Jupyter Notebooks to select optimal hyperparameters for Q-Learning:
    1. Q-Learning Easy.ipynb
    2. Q-Learning Hard.ipynb
4. Update the following notebooks based on chosen optimal hyperparameters for Q-Learning. Run Jupyter Notebooks to produce comparison charts between the three algorithms. 
    1. Easy Comparison - Policy, Value, Q-Learning.ipynb
    2. Hard Comparison - Policy, Value, Q-Learning.ipynb
