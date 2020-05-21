import numpy as np
import warnings
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
import graphviz
import os
import traceback
import time
from Customization import exponent, power
from DataExtraction import create_dataset
os.environ["PATH"] += os.pathsep + 'C:\\Users\\blued\\Desktop\\Self Learn\\Virtual Environments\\LGML\\graphviz-2.38\\release\\bin\\'



def gplearn_procedure(equation_id, no_samples=1000, input_range=(-1, 1), save_path=None, save=True, load=True, func_set=['add', 'sub', 'mul', 'div', 'log', 'sqrt', 'cos', 'tan', 'sin', 'pow', 'exp'], verbose=1):
    """
    Uses gplearn to attempt to predict the equation form of 'equation_id'
    Renders a graphviz image to images/gplearn/
    returns predicted equation, R^2 score and time taken
    
    Parameters
    ----------
    equation_id : string
        The ID of an equation in the dataset. Must be a valid one

    no_samples : int 
        The number of samples you want fed in to the algorithm

    input_range: tuple(float, float)
        The minimum and maximum values of all input parameters
    save_path: string path
        The path to where you wish the save this dataframe
    save: boolean
        Saves file to save_path iff True
    load: boolean
        If true then looks for file in save_path and loads it preemptively if it is there

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

    verbose : int
        controls how much is printed, 0 is quitest

    Returns
    -------
    string, float, float
    """
    try:
        df = create_dataset(equation_id, no_samples = no_samples,input_range=input_range, save_path=save_path, save=save, load=load).dropna()
        X = df.drop('target', axis=1)
        y = df['target']
    except:
        traceback.print_exc()
        print(f"Error on equation {equation_id} skipping")
        return '', 0, 0
    no_samples = min(no_samples, len(y))
    
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

    est_gp = SymbolicRegressor(population_size=5000,
                               generations=10, stopping_criteria=0.01,
                               p_crossover=0.7, p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05, p_point_mutation=0.1,
                               max_samples=0.9, function_set=final_func_set,  verbose=verbose,
                               parsimony_coefficient=0.01, random_state=0)

    start = time.time()
    hist = est_gp.fit(X[:no_samples], y[:no_samples])
    end = time.time()
    #print(est_gp._program)
    dot_data = est_gp._program.export_graphviz()
    graph = graphviz.Source(dot_data)
    graph.render(f'images/gplearn/{equation_id}_estimate', format='png', cleanup=True)
    return est_gp._program, est_gp.score(X, y) ,end-start
