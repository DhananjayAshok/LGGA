from LearningSystems.GPLearnSystem import GPLearnSystem
from LearningSystems.DEAPLearningSystem import DEAPLearningSystem
from Trainer import Trainer


from LearningSystems.DEAPLearningSystem import triangle_rule

func_set = ['add', 'mul', 'sub', 'div', 'exp', 'sqrt', 'sin', 'cos', 'tan', 'pow']

trainer = Trainer(path="data//", save=True, load=True, noise_range=(-0.025, 0.025), master_file="OtherEquations.csv")
gp = GPLearnSystem(func_set=func_set)
dl = DEAPLearningSystem(func_list=func_set)
dl.set_add_func(triangle_rule)
print(trainer.predict_equations(dl, no_samples=10, eqs=["pythogoras"], input_range=(-200, 200)))
