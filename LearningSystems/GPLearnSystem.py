from .LearningSystem import LearningSystem
import warnings
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from gplearn.fitness import make_fitness
import graphviz
import os
from Customization import *

class GPLearnSystem(LearningSystem):
    def __init__(self, func_set=['add', 'sub', 'mul', 'div', 'log', 'sqrt', 'cos', 'tan', 'sin', 'pow', 'exp'], path="gplearn_data"):
        """
        Sets the final_func set after using the appropriate gplearn methods

        Parameters
        -----------
            func_set : list
                List of strings i.e names of functions to include / operations to consider
                current options include
                ‘add’ : addition, arity=2.
                ‘sub’ : subtraction, arity=2.
                ‘mul’ : multiplication, arity=2.
                ‘div’ : protected division where a denominator near-zero returns 1., arity=2.
                ‘sqrt’ : protected square root where the absolute value of the argument is used, arity=1.
                ‘log’ : protected log where the absolute value of the argument is used and a near-zero argument returns 0., arity=1.
                ‘abs’ : absolute value, arity=1.
                ‘neg’ : negative, arity=1.
                ‘inv’ : protected inverse where a near-zero argument returns 0., arity=1.
                ‘max’ : maximum, arity=2.
                ‘min’ : minimum, arity=2.
                ‘sin’ : sine (radians), arity=1.
                ‘cos’ : cosine (radians), arity=1.
                ‘tan’ : tangent (radians), arity=1.

                'exp' : exponential (self defined), arity=1
                'pow' : power (self defined), arity=2

            path : str
                path to save data logs and images
        """
        default_func_set = ('add', 'sub', 'mul', 'div', 'log', 'sqrt', 'cos', 'tan', 'sin', 'abs', 'neg', 'inv', 'max', 'min' )
        final_func_set = []
        for func in func_set:
            if func in default_func_set:
                final_func_set.append(func)
            else:
                if func == "pow":
                    final_func_set.append(make_function(power, func, 2))
                elif func == "exp":
                    final_func_set.append(make_function(exponent, func, 1))
                elif func == "pi":
                    final_func_set.append(make_function(pi, func, 0))
                else:
                    warnings.warn(f"{func} is an unrecognized function, skipping it")
                    pass
        self.final_func_set = final_func_set
        self.path = path
        return

    def fit(self, X, y):
        """
        Performs a fit function
        """
        self.est_gp = SymbolicRegressor(population_size=5000,
                               generations=10, stopping_criteria=0.01,
                               p_crossover=0.7, p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05, p_point_mutation=0.1,
                               max_samples=0.9, function_set=self.final_func_set,  verbose=0,
                               parsimony_coefficient=0.01, random_state=0)
        return self.est_gp.fit(X, y)


    def score(self, X, y):
        """
        R^2
        """
        return self.est_gp.score(X, y)

    def get_predicted_equation(self):
        return self.est_gp._program

    def __str__(self):
        return "gplearn"
        

def sine_constraint_fitness(y, y_pred, w, delta=1):
    """
    Using auxilliary truth sin^2 + cos^2 = 1
    """
    mse = np.average((y-y_pred)**2, weights=w)
    truth_diff = delta * y_pred 



