from LearningSystems.GPLearnSystem import GPLearnSystem
from LearningSystems.DEAPLearningSystem import DEAPLearningSystem
from Trainer import Trainer
import pandas as pd
from Constraints import *

equation_list = ["resistance", "snell", "coloumb", "reflection", "gas", "distance", "normal"]
basic_func_set = ["add", "mul", "sub", "div"]
snell_func_set = basic_func_set + ["sin"]
reflection_func_set = basic_func_set + ["abs", "square"]
distance_func_set = basic_func_set + ["square", "sqrt"]

eqs = ["resistance", "snell", "coloumb", "reflection"]
func_dict = {"snell": snell_func_set,"coloumb":basic_func_set, "reflection":reflection_func_set, "gas":basic_func_set, "distance":distance_func_set}
weight_dict = {"snell":0.25, "coloumb":9_000,"reflection":1.2,"gas":1_00_000, "distance":10}
constraints_dict = {"snell":snell_constraints, "coloumb":coloumb_constraints, "reflection":reflection_constraints, "gas":gas_constraints, "distance":distance_constraints}
lgml_dict = {"snell":snell_lgml_func, "coloumb":coloumb_lgml_func, "reflection":reflection_lgml_func, "gas":gas_lgml_func, "distance":distance_lgml_func}
size_dict = {"snell":150, "coloumb":1_200,"reflection":1_500,"gas":250, "distance":1600}


def equation_report_lgml(eq, func_dict, weight_dict, constraints_dict, lgml_dict, nruns=15):
    func_set = func_dict[eq]
    tweight = weight_dict[eq]
    constraints_func = constraints_dict[eq]
    lgml_func = lgml_dict[eq]

    trainer = Trainer(path="data//", save=True, load=True, master_file="OtherEquations.csv")
    dl = DEAPLearningSystem(func_list=func_set, ngens=15, algorithm="lgml", population_size=50)
    weightlist = [tweight for i in range(nruns)]
    no_examples = 1
    for i in range(no_examples):
        data =[]
        for weight in weightlist:
            dl.set_add_func(lambda dls, x, y : constraints_func(dls, x, y, weight=weight))
            dl.set_lgml_func(lgml_func)
            df = trainer.predict_equations(dl, no_train_samples=150, eqs=[eq], use_gens=True)
            temp = df.loc[0, :]
            temp["Weight"] = weight
            temp["MSE"] = temp["Error"][0]
            temp["Truth Error"] = temp["Error"][1]
            temp = temp.reindex(index=['Weight', 'Predicted Equation', 'MSE',"Truth Error", 'Time Taken'])
            data.append(temp)
        final_df = pd.DataFrame(data)
        final_df.set_index("Weight")
        final_df.to_csv(f"{eq}LGML{i}.csv", index=False)


def equation_report_baseline(eq, func_dict, weight_dict, constraints_dict, size_dict, nruns=15):
    func_set = func_dict[eq]
    tweight = weight_dict[eq]
    constraints_func = constraints_dict[eq]
    nsamples = size_dict[eq]

    trainer = Trainer(path="data//", save=True, load=True, master_file="OtherEquations.csv")
    dl = DEAPLearningSystem(func_list=func_set, ngens=15, algorithm="custom", population_size=50)
    weightlist = [0 for i in range(nruns)]
    no_examples = 1
    for i in range(no_examples):
        data =[]
        for weight in weightlist:
            dl.set_add_func(lambda dls, x, y : constraints_func(dls, x, y, weight=weight))
            #dl.set_lgml_func(lgml_func)
            df = trainer.predict_equations(dl, no_train_samples=nsamples, eqs=[eq], use_gens=True)
            temp = df.loc[0, :]
            temp["Weight"] = weight
            temp["MSE"] = temp["Error"][0]
            temp["Truth Error"] = temp["Error"][1]
            temp = temp.reindex(index=['Weight', 'Predicted Equation', 'MSE',"Truth Error", 'Time Taken'])
            data.append(temp)
        final_df = pd.DataFrame(data)
        final_df.set_index("Weight")
        final_df.to_csv(f"{eq}Baseline{i}.csv", index=False)



for eq in []:
    print(f"\n\nNow Starting LGML for Equation {eq}\n\n")
    equation_report_lgml(eq, func_dict, weight_dict, constraints_dict, lgml_dict)
    print(f"\n Finished LGML Run for Equation {eq}\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n")

for eq in ['gas', 'distance']:
    equation_report_baseline(eq, func_dict, weight_dict, constraints_dict, size_dict)

