from LearningSystems.GPLearnSystem import GPLearnSystem
from LearningSystems.DEAPLearningSystem import DEAPLearningSystem
from Trainer import Trainer
import pandas as pd
from Constraints import *

equation_list = ["resistance", "snell", "coloumb", "reflection", "gas", "distance", "normal"]
feynman_equations = ["I.11.19", "I.12.11", "I.13.12", "I.18.4", "I.18.14","I.34.1", "I.39.11","I.44.4", "I.47.23", "II.34.11"]
basic_func_set = ["add", "mul", "sub", "div"]
snell_func_set = basic_func_set + ["sin"]
reflection_func_set = basic_func_set + ["abs", "square"]
distance_func_set = basic_func_set + ["square", "sqrt"]
I444_func_set = basic_func_set + ['log']

eqs = [ "coloumb", "reflection", "gas"]
func_dict = {"resistance": basic_func_set, "snell": snell_func_set,"coloumb":basic_func_set, "reflection":reflection_func_set, "gas":basic_func_set, "distance":distance_func_set, "I.11.19":basic_func_set, "I.12.11":basic_func_set, "I.13.12":basic_func_set, "I.18.4":basic_func_set, "I.18.14":snell_func_set,"I.34.1":basic_func_set, "I.39.11":basic_func_set,"I.44.4":I444_func_set, "I.47.23":basic_func_set, "II.34.11":basic_func_set}
weight_dict = {"resistance": 0.25, "snell":0.25, "coloumb":9_000,"reflection":1.2,"gas":1_00_000, "distance":10, "I.11.19":0.25, "I.12.11":0.25, "I.13.12":0.25, "I.18.4":0.25, "I.18.14":0.25,"I.34.1":0.25, "I.39.11":0.25,"I.44.4":0.25, "I.47.23":0.25, "II.34.11":0.25}
constraints_dict = {"resistance": resistance_constraints, "snell":snell_constraints, "coloumb":coloumb_constraints, "reflection":reflection_constraints, "gas":gas_constraints, "distance":distance_constraints, "I.11.19":I1119_constraints, "I.12.11":I1211_constraints, "I.13.12":I1312_constraints, "I.18.4":I184_constraints, "I.18.14":I1814_constraints,"I.34.1":I1814_constraints, "I.39.11":I3911_constraints,"I.44.4":I444_constraints, "I.47.23":I4723_constraints, "II.34.11":II3411_constraints}
lgml_dict = {"resistance": resistance_lgml_func, "snell":snell_lgml_func, "coloumb":coloumb_lgml_func, "reflection":reflection_lgml_func, "gas":gas_lgml_func, "distance":distance_lgml_func, "I.11.19":I1119_lgml_func, "I.12.11":I1211_lgml_func, "I.13.12":I1312_lgml_func, "I.18.4":I184_lgml_func, "I.18.14":I1814_lgml_func,"I.34.1":I1814_lgml_func, "I.39.11":I3911_lgml_func,"I.44.4":I444_lgml_func, "I.47.23":I4723_lgml_func, "II.34.11":II3411_lgml_func}
size_dict = {"resistance": 700, "snell":150, "coloumb":1_200,"reflection":1_500,"gas":250, "distance":1600, "I.11.19":500, "I.12.11":500, "I.13.12":500, "I.18.4":500, "I.18.14":500,"I.34.1":500, "I.39.11":500,"I.44.4":500, "I.47.23":500, "II.34.11":500}


def equation_report_lgml(eq, func_dict, weight_dict, constraints_dict, lgml_dict, nruns=10):
    func_set = func_dict[eq]
    tweight = weight_dict[eq]
    constraints_func = constraints_dict[eq]
    lgml_func = lgml_dict[eq]

    trainer = Trainer(path="data//", save=True, load=True, master_file="FeynmanEquations.csv")
    dl = DEAPLearningSystem(func_list=func_set, ngens=15, algorithm="lgml", population_size=50)
    weightlist = [tweight for i in range(nruns)]
    no_examples = 1
    for i in range(no_examples):
        data =[]
        for weight in weightlist:
            dl.set_add_func(lambda dls, x, y : constraints_func(dls, x, y, weight=weight))
            dl.set_lgml_func(lgml_func)
            df = trainer.predict_equations(dl, no_train_samples=150, eqs=[eq], use_gens=False)
            temp = df.loc[0, :]
            temp["Weight"] = weight
            temp["MSE"] = temp["Error"][0]
            temp["Truth Error"] = temp["Error"][1]
            temp = temp.reindex(index=['Weight', 'Predicted Equation', 'MSE',"Truth Error", 'Time Taken'])
            data.append(temp)
        final_df = pd.DataFrame(data)
        final_df.set_index("Weight")
        final_df.to_csv(f"{eq}LGML{i}.csv", index=False)


def equation_report_baseline(eq, func_dict, weight_dict, constraints_dict, size_dict, nruns=10):
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

def equation_report_early_switching(eq, func_dict, weight_dict, constraints_dict, size_dict, nruns=10):
    func_set = func_dict[eq]
    tweight = weight_dict[eq]
    constraints_func = constraints_dict[eq]
    nsamples = size_dict[eq]

    trainer = Trainer(path="data//", save=True, load=True, master_file="OtherEquations.csv")
    dl = DEAPLearningSystem(func_list=func_set, ngens=15, algorithm="earlyswitcher", population_size=50)
    weightlist = [tweight for i in range(nruns)]
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
        final_df.to_csv(f"{eq}EarlySwitching{i}.csv", index=False)



for eq in feynman_equations:
    print(f"\n\nNow Starting LGML for Equation {eq}\n\n")
    equation_report_lgml(eq, func_dict, weight_dict, constraints_dict, lgml_dict, nruns=2)
    print(f"\n Finished LGML Run for Equation {eq}\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n")

for eq in []:
    equation_report_baseline(eq, func_dict, weight_dict, constraints_dict, size_dict)

for eq in []:
    print(f"\n\nNow Starting Early Switching for Equation {eq}\n\n")
    equation_report_early_switching(eq, func_dict, weight_dict, constraints_dict, size_dict)
    print(f"\n Finished ES Run for Equation {eq}\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n")