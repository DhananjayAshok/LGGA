from LearningSystems.GPLearnSystem import GPLearnSystem
from LearningSystems.DEAPLearningSystem import DEAPLearningSystem
from Trainer import Trainer
from Constraints import *

equation_list = ["resistance", "snell", "coloumb", "reflection", "gas", "distance", "normal"]
func_set = ['add', 'mul', 'sub', 'div', 'square', 'sqrt']

trainer = Trainer(path="data//", save=True, load=True, noise_range=(-0.025, 0.025), master_file="OtherEquations.csv")
dl = DEAPLearningSystem(func_list=func_set, ngens=15, algorithm="custom", population_size=50)
import pandas as pd
weightlist = [0.25+(0.5*i) for i in range(5)]
no_examples = 1
for i in range(no_examples):
    data =[]
    for weight in weightlist:
        dl.set_add_func(lambda dls, x, y : snell_constraints(dls, x, y, weight=weight))
        #dl.set_lgml_func(distance_lgml_func)
        df = trainer.predict_equations(dl, no_train_samples=500, eqs=["snell"], input_range=(0, 100), use_gens=True)
        temp = df.loc[0, :]
        temp["Weight"] = weight
        temp["MSE"] = temp["Error"][0]
        temp["Truth Error"] = temp["Error"][1]
        temp = temp.reindex(index=['Weight', 'Predicted Equation', 'MSE',"Truth Error", 'Time Taken'])
        data.append(temp)
    final_df = pd.DataFrame(data)
    final_df.set_index("Weight")
    final_df.to_csv(f"SnellWeights{i}.csv", index=False)

# f(x)^2 + ((f(x -eps) + f(x + eps) / 2 * eps)^2 == 1