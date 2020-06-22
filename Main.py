from LearningSystems.GPLearnSystem import GPLearnSystem
from LearningSystems.DEAPLearningSystem import DEAPLearningSystem
from Trainer import Trainer
from Constraints import *

func_set = ['add', 'mul', 'sub', 'div', 'exp', 'sqrt', 'pow']

trainer = Trainer(path="data//", save=True, load=True, noise_range=(-0.025, 0.025), master_file="OtherEquations.csv")
#gp = GPLearnSystem(func_set=func_set)
dl = DEAPLearningSystem(func_list=func_set, ngens=15, algorithm="custom")
import pandas as pd
sizelist = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
sizelist = [100]
no_examples = 1
for i in range(no_examples):
    data =[]
    for size in sizelist:
        dl.set_add_func(lambda dls, x, y : triangle_rule(dls, x, y, weight=1000000))
        df = trainer.predict_equations(dl, no_train_samples=size, eqs=None, input_range=(0, 500), use_gens=True)
        temp = df.loc[0, :]
        temp["Size"] = size
        temp["MSE"] = temp["Error"][0]
        temp["Truth Error"] = temp["Error"][1]
        temp = temp.reindex(index=['Size', 'Predicted Equation', 'MSE',"Truth Error", 'Time Taken'])
        data.append(temp)
    final_df = pd.DataFrame(data)
    final_df.set_index("Size")
    final_df.to_csv(f"TriangleAnalysis{i}.csv", index=False)

# f(x)^2 + ((f(x -eps) + f(x + eps) / 2 * eps)^2 == 1