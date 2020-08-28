from DataExtraction import create_dataset
import traceback
import pandas as pd
from tqdm import tqdm
import time
from Equations import equation_dict
from Generators import *

class Trainer(object):
    """
    Class responsible for handling the loading and training of data overall
    """
    def __init__(self, path=None, save=False, load=False, noise_range=(0, 0), master_file="FeynmanEquations.csv"):
        """
        Input parameters decide whether or not the dataset created by the trainer should be loaded from saved data (load) or saved after creation (save) and if so from where/ where.

        noise_range: tuple(float, float)
            The minimum and maximum value of normally distributed noise added to the data

        master_file : str
            The path to the master file of equations. This csv file must at least contain the following columns - "Filename"(equation ids), "Formula"(string readable), "nvariables(int readable that matches number of variables of the equation on that line)"
        """
        self.path = path
        self.save = save
        self.load = load
        self.noise_range= noise_range
        self.master_file = master_file
        return

    def set_path(self, path):
        self.path = path

    def set_laod(self, load):
        self.load = load

    def set_save(self, save):
        self.save = save

    def set_master_file(self, master_file):
        self.master_file = master_file

    def set_noise_range(self, noise_range):
        self.noise_range = noise_range

    def predict_single_equation(self, equation_id, learning_system, no_train_samples=1000, no_test_samples=1000, input_range=(-100, 100), use_gens=False):
        """
        Returns predicted equation, error metric and time taken for fitting
        
        Parameters
        ----------
        equation_id : string
            The ID of an equation in the dataset. Must be a valid one

        learning_system : System
            An instance of a System class that implements __str__, fit, get_predicted_equation and score

        no_samples : int 
            The number of samples you want fed in to the algorithm

        input_range: tuple(float, float)
            The minimum and maximum values of all input parameters


        Returns
        -------
        string, float, float
        """
        try:
            df = create_dataset(equation_id, no_samples = no_test_samples,input_range=input_range, path=self.path, save=self.save, load=self.load, noise_range=self.noise_range, master_file=self.master_file).dropna()
            X = df.drop('target', axis=1)
            y = df['target']
        except:
            traceback.print_exc()
            print(f"Error on equation {equation_id} skipping")
            return '', 0, 0
        if not use_gens:
            no_samples = min(no_train_samples, len(y))
            start = time.time()
            try:
                hist = learning_system.fit(X[:no_samples], y[:no_samples], equation_id=equation_id)
            except:
                traceback.print_exc()
                print(f"Error Fitting Learning System {learning_system} on equation {equation_id}")
                return '', 0, 0
            end = time.time()
        else:
            gen = None
            if equation_id in equation_dict.keys():
                gen = get_generator_generic(equation_id, no_samples=no_train_samples, input_range=input_range, noise_range=noise_range, master_file=master_file)
            else:
                if equation_id not in generator_dict.keys():
                    print(f"Equation {equation_id} does not have a generator registered in generator_dict. Please register the equation to run with use_gens parameter. Current generator_dict keys : \n{generator_dict.keys()}")
                    return '', 0, 0
                gen = generator_dict.get(equation_id)(no_samples=no_train_samples, input_range=input_range)
            start = time.time()
            try:
                hist = learning_system.fit_gen(gen)
            except:
                traceback.print_exc()
                print(f"Error Fitting Learning System {learning_system} on equation {equation_id}")
                return '', 0, 0
            end = time.time()

        return learning_system.get_predicted_equation(), learning_system.score(X, y) , end-start


    def predict_equations(self, learning_system, eqs=None, save_every=15, no_train_samples=1000, no_test_samples=1000, input_range=(-100, 100), use_gens=False):
        """
        Creates and Returns a DataFrame with columns real_equation, predicted_equation, error metric, time taken (s)
        Also saves this DataFrame to the path variable of this LearningSystem object.
        
        If save is true saves the data points generated from the equations to a csv file in the path variable of this Trainer object.
        If load is true attempts to load the data file from memory instead of generating it (attempts to load from save_path
        Uses the master_file to guide the loading of equations and number of variables
        Save, load, path and master_file can be changed with setter methods that are provided or set during initialization

        path variable of the LearningSystem can be set with initializer of the LearningSystem
        
        Parameters
        ----------
        learning_system : System
            An instance of a System class that implements __str__, get_path , fit, get_predicted_equation and score

        eqs : Optional[None, int, Iterable[String]]
            If None function predicts for all available equations| if int then the function predicts for the first eqs equations | else must be a collection of valid equation_ids

        save_every: int
            Every save_every iterations the results will be saved to a different DataFrame    

        no_samples : int 
            The number of samples you want fed in to the algorithm

        input_range: tuple(float, float)
            The minimum and maximum values of all input parameters

        Returns
        -------
        DataFrame

        """
        logs = []
        global_logs = []
        master_data = pd.read_csv(self.master_file)
        read_eqs = master_data['Filename']
        if eqs is None:
            eqs = read_eqs
            neqs = len(eqs)
        elif type(eqs) == int:
            neqs = eqs
            eqs = read_eqs
        else:
            teqs = list(set(eqs).intersection(set(read_eqs)))
            if len(teqs) == 0:
                raise ValueError(f"None of the equations provided in the iterable {teqs}")
            else:
                eqs = teqs
                neqs = len(eqs)

        n = 0
        for i in tqdm(range(neqs)):
            eq = eqs[i]
            equation, error, time = self.predict_single_equation(eq, learning_system, no_train_samples=no_train_samples, no_test_samples=no_test_samples, input_range=input_range, use_gens=use_gens)
            logs.append([eq, master_data['Formula'][i], equation, error, time])
            global_logs.append([eq, master_data['Formula'][i], equation, error, time])
            n += 1
            if n % save_every == 0 or n == neqs:
                df = pd.DataFrame(logs, columns=["Equation_ID", "Real Equation", "Predicted Equation", "Error", "Time Taken"])
                df.to_csv(f'{learning_system.get_path()}//{str(learning_system)}_logs{n-save_every}-{n}.csv',index=False )
                logs = []
        df = pd.DataFrame(global_logs, columns=["Equation_ID", "Real Equation", "Predicted Equation", "Error", "Time Taken"])
        df.to_csv(f'{learning_system.get_path()}//{str(learning_system)}_global_logs.csv', index=False)
        return df







