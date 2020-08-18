# [Logic Guided Genetic Algorithms for Symbolic Regression]()
This is our implementation of the Logic Guided Genetic Algorithm, a genetic algorithm for Symbolic Regression that combines logical constraints with the genetic programming life cycle to help avoid local minima and perform effective data set augmentation.

This code was written by [Dhananjay Ashok](https://www.linkedin.com/in/dhananjay-ashok-576342142/), who is also the lead author on the working paper on the same topic.

## Prerequisites
* Python 3.6+

## Setup
Install required python packages, if they are not already installed:
``` bash
pip install requirements.txt
```


Clone this repository:
``` bash
git clone https://github.com/DhananjayAshok/Genetic-Algorithms-with-Logic-Guided-Machine-Learning LGGA
cd LGGA
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
1. Add a Generator Producer Function which returns a generator for the equation (Examples of this seen in Generators.py)
2. Add this Generator Producer Function to the generator_dict of the Generators.py file
3. Add any constraints relevant to the equation in Constraints.py (format and convinience functions can be found in Constraints.py)
4. Add any data generation (lgml_func) function which may apply to the equation
5. Make sure that there is a dataset of X, y values of this equation in the data/ folder (This can be done by running the generator once)
6. Add the equation and its details to a master File of your choosing (current master files include FeynmanEquations.csv, OtherEquations.csv)

To run the LGGA Tool on the new equation after set 
1. Create a Trainer object and specify which master file the equation is in
2. Create a DEAPLearningSystem Object and specify which algorithm you wish to use (use lgml or earlyswitcher for the two algorithms found in the paper)
3. Supply the constraint function and lgml functions to the DEAPLearningSystem via set_add_func and set_lgml_func
4. Call on the Trainers predict_equations with the eq specified as required and specify use_gens to be true
5. Result will be saved in DEAP_data by default 


 ## Data Set Augmentation

To create an augmented data set from the provided generators:
- Follow all the steps above 
- In line 724 in DEAPLearningSystem.py change False to True
- Optimal Dataset will now be saved in the root folder at the end of training


## Citation
If you use our work, please cite our paper. Will add link upon publication

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/). For any commercial uses or derivatives, please contact us.
