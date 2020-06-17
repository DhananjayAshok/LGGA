from LearningSystems.GPLearnSystem import GPLearnSystem
from LearningSystems.DEAPLearningSystem import DEAPLearningSystem
from Trainer import Trainer


from LearningSystems.DEAPLearningSystem import triangle_rule, semiperimeter_rule

func_set = ['add', 'mul', 'sub', 'div', 'exp', 'sqrt', 'pow']

trainer = Trainer(path="data//", save=True, load=True, noise_range=(-0.025, 0.025), master_file="OtherEquations.csv")
#gp = GPLearnSystem(func_set=func_set)
dl = DEAPLearningSystem(func_list=func_set, ngens=15)
import pandas as pd
weightlist = [0, 1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
no_examples = 5
for i in range(no_examples):
    data =[]
    for weight in weightlist:
        dl.set_add_func(lambda dls, x, y : semiperimeter_rule(dls, x, y, weight=weight))
        df = trainer.predict_equations(dl, no_samples=100, eqs=None, input_range=(-200, 200))
        temp = df.loc[0, :]
        temp["Weight"] = weight
        temp = temp.reindex(index=['Weight', 'Predicted Equation', 'Error', 'Time Taken'])
        data.append(temp)
    final_df = pd.DataFrame(data)
    final_df.set_index("Weight")
    final_df.to_csv(f"TriangleAnalysis{i}.csv", index=False)

# f(x)^2 + ((f(x -eps) + f(x + eps) / 2 * eps)^2 == 1