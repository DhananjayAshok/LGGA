# Logic Guided Genetic Algorithms

This is our implementation of the Logic Guided Genetic Algorithm, a genetic algorithm for Symbolic Regression that combines logical constraints with the genetic programming life cycle to help avoid local minima and perform effective data set augmentation.

This code was written by [Dhananjay Ashok](https://dhananjay-ashok.webnode.com/), who is also the lead author on the paper on the same topic. The other authors of the paper are - Joseph Scott, Sebastian Wetzel, Maysum Panju and Vijay Ganesh

## Prerequisites
* Python 3.6+

Clone this repository:
``` bash
git clone https://github.com/DhananjayAshok/Genetic-Algorithms-with-Logic-Guided-Machine-Learning LGGA
cd LGGA
```
## Setup
Install required python packages, and create required directories:
``` bash
bash setup.sh
```

Currently Supported Equations:
- All Equations from the [Feynman Equations Dataset](https://space.mit.edu/home/tegmark/aifeynman.html)
- Pythogorean Formula
- Parallel Resistance
- Reflectivity
- Coloumb's Law
- Snell's Law
- Gas Law
- Distance
- Normal Distribution

To add a new equation follow the steps below
1. Create an initial data set of X, y in data/equation_id.csv (all columns must be X0, X1, ... target)
3. Add any constraints relevant to the equation in Constraints.py (format and convinience functions can be found in Constraints.py)
4. Add any data generation (lgml_func) function which may apply to the equation
6. Add the equation and its details to a master File of your choosing (current master files include FeynmanEquations.csv, OtherEquations.csv)

To run the LGGA Tool on the new equation after set 
1. Create a Trainer object and specify which master file the equation is in
2. Create a DEAPLearningSystem Object and specify which algorithm you wish to use (use lgml for the two algorithm found in the paper)
3. Supply the constraint function and lgml functions to the DEAPLearningSystem via set_add_func and set_lgml_func
4. Call on the Trainers predict_equations with the eq specified as required and specify use_gens to be False
5. Result will be saved in DEAP_data by default 
6. Augmented Data Set will be saved in "Datasets/equation_id LGGA Dataset.csv"



## Citation
If you use our work, please cite our paper. Will add link upon publication

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/). For any commercial uses or derivatives, please contact us.
