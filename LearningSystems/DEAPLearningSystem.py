from inspect import signature
import operator
import traceback
import warnings
warnings.filterwarnings("ignore")


from LearningSystems.LearningSystem import LearningSystem
from Customization import *

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from deap import gp
import numpy as np

class DEAPLearningSystem(LearningSystem):
    """
    Learning Algorithm that implements the DEAP Python Library
    """
    def __init__(self, path="DEAP_data", verbose=False, population_size=100, crossover_prob=0.4, mutation_prob=0.4, ngens=30, func_list=['add', 'mul', 'sub', 'div', 'sin', 'cos', 'tan', 'exp', 'sqrt']):
        """
        Parameters
        -----------
        path : str
            Location to save graphs and data

        verbose : boolean
            True iff you want to see verbose fit for DEAP

        population_size : int
            The number of equations we want to generate every generation

        crossover_prob : float
            Probability that we crossover two equations randomly

        mutation_prob : float
            Probability that we mutate an equation randomly

        ngens : int
            Number of generations we wish to train for

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

        """
        LearningSystem.__init__(self)
        self.toolbox = base.Toolbox()
        self.path = path
        self.verbose = verbose
        self.population_size=population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.ngens = ngens
        self.func_list = func_list
        self.addfunc = self.zero
        self.creator = creator

    def create_fitness(self):
        """
        Creates a Fitness function and registers it with the toolbox as DEAP library requires
        Assumes self.toolbox is defined

        Currently minimizes a single objective function
        """
        self.creator.create("Fitness", base.Fitness, weights=(-1.0,))
        self.Fitness = self.creator.Fitness
        return

    def create_and_reg_individual(self, problem_arity):
        """
        Creates an individual and registers it with the toolbox as DEAP library requires. 
        Assumes self.create_fitness() has been called

        Currently creates an individual that inherits from PrimitiveTree and uses the primitives mentioned in list just as the default DEAP examples

        Parameters
        -----------
        problem_arity : int
            The number of input variables for this symbolic regression problem
        """

        self.pset = gp.PrimitiveSet("MAIN", arity=problem_arity)
        for func in self.func_list:
            self.pset.addPrimitive(func_dict[func], len(signature(func_dict[func]).parameters), name=func)

        self.creator.create("Individual", gp.PrimitiveTree, fitness=self.Fitness,
                       pset=self.pset)
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2)
        self.toolbox.register("individual", tools.initIterate, self.creator.Individual,
                         self.toolbox.expr)
        return

    def create_and_reg_population(self):
        """

        Currently uses bad model
        """
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        return

    def reg_selection(self, tournsize):
        """
        Registers our method of selection.

        Parameters
        -----------
        tournsize : int
            How many equations to select for every gen
        """
        self.toolbox.register("select", tools.selTournament, tournsize=tournsize)
        return

    def reg_mutation(self):
        """
        Controls how equations mutate
        """
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        return
       
    def reg_mating(self):
        """
        Controls how e cross species
        """
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        return

    def reg_eval(self, X, y):
        """
        registered the evaluation method we wanna use in this function

        Currently uses mean squared error but this is what we will have to or want to change to bring in LGML
        
        Parameters
        ------------
        X, y - Data columns and target series
        """
        def eval(ind, X, y):
            self.func = gp.compile(ind, self.pset)
            mse = self._mse(self.func, X, y)
            return mse
        self.toolbox.register('evaluate', eval, X=X , y=y)
        return

    def _mse(self, func, X, y):
        """
        Returns the mean square error of a function which can compute the value of f(X)
        """
        def temp(row):
                try:
                    val = func(*row)
                    #print(f"Value {val} was succesfully calculated for {str(expr)}")
                    return val
                except:
                    traceback.print_exc()
                    return 9999
        X['result'] = X.copy().apply(lambda row: temp(row), axis=1)
        diff = X['result'] - y
        X.drop('result', axis=1, inplace=True)
        #print(np.mean(diff*2))
        return (np.mean(diff**2), )

    def get_arity_from_X(self, X):
        return len(X.columns)

    def zero(object, func, X, y):
        return np.zeros(y.shape)
    #########################################################################
    
    def set_add_func(func):
        """
        Set additional process.
        func must be a function that takes in 4 parameters - object, func, X, y and returns an array of values of shape 
        """
        self.add_func = func

    def reset(self):
        """
        Clears all working data so it was as if this object was a newly created DEAPLearnSystem immediately after initialization
        """
        self.toolbox = base.Toolbox()
        try:
            del self.Fitness
            del self.pset
            del self.hof
        except:
            print("Already Reset")
        return

    def build_model(self, X, y, tournsize=3):
        """
        Runs the builder functions in order
        """
        arity = self.get_arity_from_X(X)
        self.create_fitness()
        self.create_and_reg_individual(arity)
        self.create_and_reg_population()
        self.reg_selection(tournsize)
        self.reg_mutation()
        self.reg_mating()
        self.reg_eval(X, y)

        return
        
    def set_func_list(self, func_list):
        """
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
        """
        self.func_list = func_list
        return

    def __str__(self):
        return "DEAP"

    def fit(self, X, y):
        """
        Currently uses a simple algorithm and returns the hall of famer
        Clears the existing trained model every time fit is called
        """
        self.reset()
        self.build_model(X, y)
        self.hof = tools.HallOfFame(1)
        pop, log = algorithms.eaSimple(self.toolbox.population(self.population_size), self.toolbox, self.crossover_prob, self.mutation_prob, self.ngens, halloffame=self.hof, verbose=self.verbose)
        return pop, log

    def get_predicted_equation(self):
        return self.toolbox.clone(self.hof[0])

    def score(self, X, y):
        """
        Returns the evaluation on this model as per its own evaluation metric

        MIGHT NEED TO CHANGE THIS TO SOMETHING STANDARD 
        """
        try:
            best = self.hof[0]
            return self.toolbox.evaluate(best)
        except:
            print(f"Could not find Best model")
            return 0