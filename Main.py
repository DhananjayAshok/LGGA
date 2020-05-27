from LearningSystems.GPLearnSystem import GPLearnSystem
from LearningSystems.DEAPLearningSystem import DEAPLearningSystem
from Trainer import Trainer


func_set = ['add', 'mul', 'sub', 'div', 'exp', 'sqrt', 'sin', 'cos', 'tan']

trainer = Trainer(path="data//", save=True, load=True, noise_range=(-0.025, 0.025), master_file="FeynmanEquations.csv")
gp = GPLearnSystem(func_set=func_set)
dl = DEAPLearningSystem(func_list=func_set)
print(trainer.predict_equations(dl, no_samples=10, eqs=50, input_range=(-200, 200)))
