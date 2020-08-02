from inspect import signature
import operator
import traceback
import warnings
import random
warnings.filterwarnings("ignore")


from LearningSystems.LearningSystem import LearningSystem
from Customization import *
from Constraints import *

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
    def __init__(self, path="DEAP_data", verbose=False, population_size=100, crossover_prob=0.3, mutation_prob=0.9, ngens=30, algorithm="simple", func_list=['add', 'mul', 'sub', 'div', 'sin', 'cos', 'tan', 'exp', 'sqrt']):
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

        algorithm: string
            Algorithm to use for training. Current options
            simple: eaSimple
            mu+lambda: eaMuPlusLambda
            mu,lambda: eaMuCommaLambda

        func_set : list
            List of strings i.e names of functions to include / operations to consider
            Check Customizations for full list

        """
        LearningSystem.__init__(self)
        self.toolbox = base.Toolbox()
        self.path = path
        self.verbose = verbose
        self.population_size=population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.ngens = ngens
        self.algorithm = algorithm
        self.func_list = func_list
        self.add_func = lambda dls, x, y : 0 # Zero Function Default
        self.lgml_func = lambda ind, dls=None, gen=None: (None, None) # Assume all true
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
        #self.pset.addTerminal(2)
        #self.pset.addTerminal(np.pi)

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
        self.toolbox.register("select", tools.selBest)
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

    def eval_helper(self, ind, X, y):
        """
        Given an X and a y returns the mse + addfunc of a given ind
        """
        self.func = gp.compile(ind, self.pset)
        mse = self._mse(self.func, X, y)
        a = self.add_func(self, X, y) 
        return (mse + a,)

    def mse_helper(self, ind, X, y):
        """
        Given an X and a y returns the mse of a given ind
        """
        self.func = gp.compile(ind, self.pset)
        mse = self._mse(self.func, X, y)
        return (mse, )

    def add_func_helper(self, ind, X, y):
        """
        Given an X and a y returns the mse of a given ind
        """
        self.func = gp.compile(ind, self.pset)
        a = self.add_func(self, X, y)
        return (a, )


    def reg_eval(self, X, y):
        """
        registered the evaluation method we wanna use in this function

        Currently uses mean squared error but this is what we will have to or want to change to bring in LGML
        
        Parameters
        ------------
        X, y - Data columns and target series
        """
        self.toolbox.register('evaluate', self.eval_helper, X=X , y=y)
        return

    def reg_gen_eval(self, generator):
        """
        registered the evaluation method using a generator

        Parameters
        --------------
        generator - a function which when called returns an X, y 
        """
        def eval(ind, gen):
            X, y = gen()
            return self.eval_helper(ind, X, y)
        self.toolbox.register('evaluate', eval, gen=generator)
        return

    def reg_mse(self, X, y):
        """
        registered the mse evaluation under toolbox.mse
        """
        self.toolbox.register('mse', self.mse_helper, X=X, y=y)
        return

    def reg_add_func(self, X, y):
        """
        registered the add_func evaluation under toolbox.addfunc
        """
        self.toolbox.register('addfunc', self.add_func_helper, X=X, y=y)
        return

    def reg_gen_mse(self, generator):
        """
        Registered the mse function using a generator
        """
        def eval(ind, gen):
            X, y = gen()
            return self.mse_helper(ind, X, y)
        self.toolbox.register('mse', eval, gen=generator)
        return

    def reg_gen_add_func(self, generator):
        """
        Registered the addfunc function using a generator
        """
        def eval(ind, gen):
            X, y = gen()
            return self.add_func_helper(ind, X, y)
        self.toolbox.register('addfunc', eval, gen=generator)
        return


    def extendX(self, addition_dataframe):
        """
        Updates X
        """
        self.X = pd.concat([self.X, addition_dataframe], ignore_index=True)
        return

    def extendy(self, additional_series):
        """
        Updates y
        """
        self.y = pd.concat([self.y, additional_series], ignore_index=True)
        return

    def initialize_lgml_functions(self, gen):
        """
        Create the variables and functions needed for LGML algorithm to operate
        """
        X, y = gen()
        self.X = X
        self.y = y
        self.toolbox.register('generate', gen)
        self.toolbox.register('getX', lambda : self.X)
        self.toolbox.register('gety', lambda : self.y)
        self.toolbox.register('extendX', self.extendX)
        self.toolbox.register('extendy', self.extendy)
        self.toolbox.register('evaluate', self.eval_helper)
        self.toolbox.register('get_violation_frame', self.lgml_func, dls=self, gen=gen)
        self.toolbox.register('compile', gp.compile, pset=self.pset)
        return
        
    def get_result(self, func, X, y):
        """
        Returns a series that holds all the values func(X)
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
        to_return = X['result'].copy()
        X.drop('result', axis=1, inplace=True)
        return to_return

    def _mse(self, func, X, y):
        """
        Returns the mean square error of a function which can compute the value of f(X)
        """
        preds = self.get_result(func, X, y)
        diff = preds - y
        return np.mean(diff**2)

    def ind_score(self, ind, X, y):
        """
        Method to score and indivudual

        Currently uses mean squared error, violations
        
        Parameters
        ------------
        X, y - Data columns and target series
        """
        self.func = gp.compile(ind, self.pset)
        mse = self._mse(self.func, X, y)
        violation = self.add_func(self, X, y)
        return (mse, violation)

    def get_arity_from_X(self, X):
        return len(X.columns)
    #########################################################################
    
    def set_add_func(self, func):
        """
        Set additional process.
        
        Parameters
        ---------------
        fun : function(deaplearningsystemobject dls, X, y) -> float
            Remember 
                you can use dls.func to get the function to get the functional transformation
                you can use dls.get_result to get the preds in a series
        """
        self.add_func = func

    def set_lgml_func(self, func):
        """
        Set the LGML style function for use
        Will only be used if algorithm is  "lgml"
        func must take in an individual precompiled with pset and return a tuple - DataFrame, Series or None, None
            must have optional parameters dls and gen
        """
        self.lgml_func = func

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm

    def reset(self):
        """
        Clears all working data so it was as if this object was a newly created DEAPLearnSystem immediately after initialization
        """
        self.toolbox = base.Toolbox()
        try:
            del self.Fitness
            del self.pset
            del self.hof
            if hasattr(self, "X"):
                del self.X
                del self.y
        except:
            print("Already Reset")
        return

    def build_model(self, X, y, tournsize=3):
        """
        Runs the builder functions in order with data
        """
        arity = self.get_arity_from_X(X)
        if self.algorithm == "lgml":
            raise ValueError(f"Trying to use algorithm lgml with a fixed X and y dataset. This is not permitted. To use LGML algorithm please call on model fit with fit_gen and provide a generator.")
        self.invariant_build_model(arity, tournsize)
        self.reg_eval(X, y)
        if self.algorithm == "earlyswitcher":
            self.reg_mse(X, y)
            self.reg_add_func(X, y)
        return

    def build_gen_model(self, generator, tournsize=15):
        """
        Runs builder functions in order with generator
        """
        smallx, smally = generator(no_samples=1)
        arity = self.get_arity_from_X(smallx)
        self.invariant_build_model(arity, tournsize)
        if self.algorithm == "lgml":
            self.initialize_lgml_functions(generator)
        else:
            self.reg_gen_eval(generator)
            if self.algorithm == "earlyswitcher":
                self.reg_gen_mse(generator)
                self.reg_gen_add_func(generator)

        return

    def invariant_build_model(self, arity, tournsize):
        self.create_fitness()
        self.create_and_reg_individual(arity)
        self.create_and_reg_population()
        self.reg_selection(tournsize)
        self.reg_mutation()
        self.reg_mating()

    def set_func_list(self, func_list):
        """
        Parameters
        -----------
        func_set : list
            Refer to Customization.py for list of functions
        """
        self.func_list = func_list
        return

    def __str__(self):
        return "DEAP"

    def train(self):
        """
        Assumes that model is fully built

        Currently uses a simple algorithm and returns the hall of famer
        Clears the existing trained model every time fit is called
        """
        self.hof = tools.HallOfFame(1)
        pop, log = get_algorithm(self.algorithm)(population=self.toolbox.population(self.population_size), toolbox=self.toolbox, cxpb=self.crossover_prob, mutpb=self.mutation_prob, ngen=self.ngens, halloffame=self.hof, verbose=self.verbose)
        return pop, log

    def fit(self, X, y):
        """
        Fit using fixed X and y
        """
        self.reset()
        self.build_model(X, y)
        return self.train()

    def fit_gen(self, gen):
        """
        Fit using generator
        """
        self.reset()
        self.build_gen_model(gen)
        return self.train()

    def get_predicted_equation(self):
        return self.toolbox.clone(self.hof[0])

    def score(self, X, y):
        """
        Returns the evaluation on this model as per its mse
        """
        try:
            best = self.hof[0]
            return self.ind_score(best, X, y)
        except:
            print(f"Could not find Best model")
            return 0



class Algorithms():
    """
    Class to hold all of the Algorithms used for DEAPLearningSystem
    All hyperparameters than are unique to a particular function are defined here.
        Not defined here - pop, toolbox, cxpb, mutpb, ngen
    """
    mu = 5
    lambda_ = 8


    def basic_self(population, toolbox, cxpb, mutpb, ngen, halloffame, verbose, population_ratio=5, early_stopping=10, threshold = 0.000000001):
        """
        Implements the following basic ideas - 
        1. Ensures the selected from the previous population are preserved if they beat the new population
        2. Implements Early Stopping if the training loss stays constant for a given number of epochs
        3. Implements target detection which terminates if error is below a certain threshold

        population_ratio should be a multiple of 2 greater than 2 and the algorithm works best when the population is exactly divisible by the ratio

        """
        early_stopping_counter = early_stopping
        best_error = threshold + 9000
        g = 0
        while (g < ngen and early_stopping_counter !=0 and best_error > threshold):
            
            # Select the next generation individuals
            selected = toolbox.select(population, len(population)//population_ratio)
            #print(f"Starting Loop on gen {g} population length is {len(population)} and selected length is {len(selected)}")
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, selected))

            # Apply crossover on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values


            # Apply mutation on the offspring
            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Create a new population and evaluate their fitnesses
            scaling = population_ratio // 2
            cohort = offspring + selected*scaling
            new_pop = toolbox.population(max(len(population) - len(cohort), 0))
            fitnesses = toolbox.map(toolbox.evaluate, new_pop)
            for ind, fit in zip(new_pop, fitnesses):
                ind.fitness.values = fit

            # The population is then re-created            
            population[:] = cohort + new_pop
            #print(f"On Gen {g}: Population: {len(population)}\n Offspring: {len(offspring)}\n Selected: {len(selected)} New_pop: {len(new_pop)}\n\n\n\n\n")

            # Get the new best error score
            new_best_error = min(tools.selBest(population, 1)[0].fitness.values[0], best_error)

            # If best error has not improved then increment early stopping metric
            if new_best_error >= best_error:
                early_stopping_counter -= 1
            else:
                early_stopping_counter = early_stopping


            # Update the best error 
            best_error = new_best_error



            # The counter is updated to indicate a next generation
            g += 1


 
        if True:
            if early_stopping_counter == 0:
                print(f"Early Stopping after {g} generations")
            elif best_error <= threshold:
                print("Threshold Reached")
        halloffame.update(population)
        return population, None

    def early_switcher(population, toolbox, cxpb, mutpb, ngen, halloffame, verbose, population_ratio=5, early_stopping=4, threshold = 0.000000001):
        """
        Implements the following basic ideas - 
        1. Ensures the selected from the previous population are preserved if they beat the new population
        2. Implements Early Stopping if the training loss stays constant for a given number of epochs
        3. Implements target detection which terminates if error is below a certain threshold

        population_ratio should be a multiple of 2 greater than 2 and the algorithm works best when the population is exactly divisible by the ratio

        """
        base_early_stoppings = [early_stopping, 1]
        early_stopping_counters = [early_stopping, base_early_stoppings[1]]
        g = 0
        best_errors = [threshold+9000, threshold+9000]
        current_evals = [toolbox.mse, toolbox.addfunc]
        current_eval = 0
        switched = False
        terminate = False
        while ((g < ngen and best_errors[0] > threshold) or (current_eval==1 and not terminate)):
            # We never want to end on a truth cycle: so if g >= ngen then we do one more mse sweep across the population and end it
            if g >= ngen:
                terminate = True
                current_eval = 0
            
            # Select the next generation individuals
            selected = toolbox.select(population, len(population)//population_ratio)
            #print(f"Starting Loop on gen {g} population length is {len(population)} and selected length is {len(selected)}")
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, selected))

            # Apply crossover on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values


            # Apply mutation on the offspring
            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            if not switched:
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            else:
                invalid_ind = offspring + selected
            fitnesses = toolbox.map(current_evals[current_eval], invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Evaluate new pop
            scaling = population_ratio // 2
            cohort = offspring + selected*scaling
            new_pop = toolbox.population(max(len(population) - len(cohort), 0))
            fitnesses = toolbox.map(current_evals[current_eval], new_pop)
            for ind, fit in zip(new_pop, fitnesses):
                ind.fitness.values = fit
            # The population is entirely replaced by the offspring
            
            population[:] = cohort + new_pop
            #print(f"On Gen {g}: Population: {len(population)}\n Offspring: {len(offspring)}\n Selected: {len(selected)} New_pop: {len(new_pop)}\n\n\n\n\n")

            # Get the new best error score
            best_performer = tools.selBest(population, 1)[0]
            new_best_error = min(best_performer.fitness.values[0], best_errors[current_eval])

            # If best error has not improved then increment early stopping metric
            if new_best_error >= best_errors[current_eval]:
                early_stopping_counters[current_eval] -= 1
            else:
                early_stopping_counters[current_eval] = base_early_stoppings[current_eval]

            # Update the best error 
            best_errors[current_eval] = new_best_error

            # The counter is updated to indicate a next generation
            g += 1

            if early_stopping_counters[current_eval] == 0:
                early_stopping_counters[current_eval] = base_early_stoppings[current_eval]
                current_eval = (current_eval+1)%2
                switched = True
            else:
                switched = False

        if True:
            if best_errors[0] <= threshold:
                print("Threshold Reached")
        halloffame.update(population)
        return population, None


    def lgml_algorithm(population, toolbox, cxpb, mutpb, ngen, halloffame, verbose, population_ratio=5, early_stopping=10, threshold = 0.0000001):
        """
        Implements the following basic ideas - 
        1. Ensures the selected from the previous population are preserved if they beat the new population
        2. Implements Early Stopping if the training loss stays constant for a given number of epochs
        3. Implements target detection which terminates if error is below a certain threshold

        """
        early_stopping_counter = early_stopping
        best_error = threshold + 9000
        g = 0
        while (g < ngen and early_stopping_counter !=0 and best_error > threshold):
            
            # Select the next generation individuals
            selected = toolbox.select(population, len(population)//population_ratio)
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, selected))

            # Apply crossover on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values


            # Apply mutation on the offspring
            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values


            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            current_X = toolbox.getX()
            current_y = toolbox.gety()
            fitnesses = toolbox.map(lambda ind : toolbox.evaluate(ind, current_X, current_y), invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Evaluate new pop
            scaling = population_ratio // 2
            cohort = offspring + selected*scaling
            new_pop = toolbox.population(max(len(population) - len(cohort), 0))
            fitnesses = toolbox.map(lambda ind : toolbox.evaluate(ind, current_X, current_y), new_pop)
            for ind, fit in zip(new_pop, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            
            population[:] = cohort + new_pop

            # Get the new best performer
            new_best = tools.selBest(population, 1)[0]
            # Get the new best error score
            new_best_error = min(new_best.fitness.values[0], best_error)

            # If best error has not improved then increment early stopping metric
            if new_best_error >= best_error:
                early_stopping_counter -= 1
            else:
                early_stopping_counter = early_stopping


            # Update the best error 
            best_error = new_best_error



            # The counter is updated to indicate a next generation
            g += 1

            # Get a pandas dataframe that returns a set of all data points where truth is violated
            violation_frameX, violation_framey = toolbox.get_violation_frame(toolbox.compile(new_best))
            if violation_frameX is None:
                pass
            else:
                try:
                    toolbox.extendX(violation_frameX)
                    toolbox.extendy(violation_framey)
                except:
                    traceback.print_exc()

                


        if True:
            if early_stopping_counter == 0:
                print(f"Early Stopping after {g} generations")
            elif best_error <= threshold:
                print("Threshold Reached")
        halloffame.update(population)
        
        print(f"Finished with {len(current_y)} Data Points")
        if True:
            toolbox.getX().to_csv('LGMLX.csv')
            toolbox.gety().to_csv('LGMLY.csv')

        return population, None
    

    def get_worst_individual_from_pop(indivuduals):
        return tools.selWorst(indivuduals, 1)



    def get_worst_individual_from_pop(indivuduals):
        return tools.selWorst(indivuduals, 1)

    



        

algo_dict = {
        "simple" : algorithms.eaSimple,
        "mu+lambda" : lambda population, toolbox, cxpb, mutpb, ngen,  halloffame, verbose : algorithms.eaMuPlusLambda(population=population, toolbox=toolbox, mu=Algorithms.mu, lambda_=Algorithms.lambda_, cxpb=cxpb, mutpb=mutpb, ngen=ngen, halloffame=halloffame, verbose=verbose),
        "mu,lambda" : lambda population, toolbox, cxpb, mutpb, ngen,  halloffame, verbose : algorithms.eaMuCommaLambda(population=population, toolbox=toolbox, mu=Algorithms.mu, lambda_=Algorithms.lambda_, cxpb=cxpb, mutpb=mutpb, ngen=ngen, halloffame=halloffame, verbose=verbose),        
        "custom"    : Algorithms.basic_self,
        "lgml"      : Algorithms.lgml_algorithm,
        "earlyswitcher": Algorithms.early_switcher
        }



def get_algorithm(key):
        """
        Returns the algorithm function associated with the key
        Defaults to eaSimple if key not found
        """
        if key not in algo_dict.keys():
            print(f"Key {key} not found out of available algorithm options. Using Simple Algorithm")
            return algorithms.eaSimple
        return algo_dict.get(key, algo_dict.get("simple"))



    

