from LearningSystems.GPLearnSystem import GPLearnSystem
from Trainer import Trainer


func_set = ['add', 'mul', 'sub', 'div', 'exp', 'sqrt', 'sin', 'cos', 'tan']

trainer = Trainer(path="data//", save=True, load=True)
gp = GPLearnSystem(func_set=func_set)
print(trainer.predict_equations(gp, eqs=2, input_range=(-200, 200)))
