---
layout: default
---

# Genetic Algorithms and Symbolic Logic
Machine Learning is a field which develops algorithms that can make it feasible to search an incredibly large space of solutions. As opposed to guaranteeing an optimal solution, many ML algorithms decide to use an indication of the fitness of a proposed solution and optimize for this value. One such class of algorithms are genetic algorithms, which iterative evolve a population of solutions, carrying forward the best of these solutions to create new more nuanced solutions. We also note the [observations](https://sites.google.com/view/logic-for-machine-learning/home?authuser=0) of other researchers: that symbolic logic, a field that regularly tries to provide guarantees of the correctness of a solution has immense potential in guiding Genetic Algorithms through their evolutionary process. 


# Logic Guided Genetic Algorithms with Symbolic Regression
This paper was written by [Dhananjay Ashok](https://dhananjay-ashok.webnode.com/), Joseph Scott, Sebastian Wetzel, Maysum Panju and Vijay Ganesh.

## Abstract
We present a novel Auxiliary Truth enhanced Genetic Algorithm (GA) that uses logical or mathematical constraints as a means of data augmentation as well as to compute loss (in conjunction with the traditional MSE), with the aim of increasing both data efficiency and accuracy of symbolic regression (SR) algorithms. Our method, logic-guided genetic algorithm (LGGA), takes as input a set of labelled data points and auxiliary truths (ATs) (mathematical facts known a priori about the unknown function the regressor aims to learn) and outputs a specially generated and curated dataset that can be used with any SR method. Three key insights underpin our method: first, SR users often know simple ATs about the function they are trying to learn. Second, whenever an SR system produces a candidate equation inconsistent with these ATs, we can compute a counterexample to prove the inconsistency, and further, this counterexample may be used to augment the dataset and fed back to the SR system in a corrective feedback loop. Third, the value addition of these ATs is that their use in both the loss function and the data augmentation process leads to better rates of convergence, accuracy, and data efficiency. We evaluate LGGA against state-of-the-art SR tools, namely, Eureqa and TuringBot on 16 physics equations from "The Feynman Lectures on Physics" book. We find that using these SR tools in conjunction with LGGA results in them solving up to 30.0% more equations, needing only a fraction of the amount of data compared to the same tool without LGGA, i.e., resulting in up to a 61.9% improvement in data efficiency.

## Results
Our experiments show us that there is significant value in combining Machine Learning and Symbolic Logic methods. We show that even state-of-the-art Symbolic Regression tools can get significantly more efficient, and discover equations with a fraction of the data when augmented using the LGGA tool. These encouraging prelimnary results have many avenues for future expansion - most notably taking inspiration from the [LGML Tool](https://arxiv.org/abs/2006.03626) and involving an SMT solver in the evolutionary cycle, to come up with the optimal points during data augmentation. 






## Citation
If you use our work, please cite our paper. [Logic Guided Genetic Algorithms](https://arxiv.org/abs/2010.11328)

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/). For any commercial uses or derivatives, please contact us.

