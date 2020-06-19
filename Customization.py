"""
Define all the symbols/ operations we plan to use here
Must be registered in the dictionary to actually work properly

The input to these functions can be either a float/ int or a 1 entry panda series

Should return a pandas series or a number doesn't matter (I think)


Currently Defined Functions
func_set : list
            List of strings i.e names of functions to include / operations to consider
            current options include
            ‘add’ : addition, arity=2.
            ‘sub’ : subtraction, arity=2.
            ‘mul’ : multiplication, arity=2.
            ‘div’ : protected division where a denominator near-zero returns 1., arity=2.
            ‘sqrt’ : protected square root where the absolute value of the argument is used, arity=1.
            ‘log’ : protected log where the absolute value of the argument is used and a near-zero argument returns 0., arity=1.
            ‘abs’ : absolute value, arity=1.
            ‘neg’ : negative, arity=1.
            ‘inv’ : protected inverse where a near-zero argument returns 0., arity=1.
            ‘max’ : maximum, arity=2.
            ‘min’ : minimum, arity=2.
            ‘sin’ : sine (radians), arity=1.
            ‘cos’ : cosine (radians), arity=1.
            ‘tan’ : tangent (radians), arity=1.
            'exp' : exponential , arity=1
            'pow' : power , arity=2
            'square': ^2, arity=1
"""


import numpy as np
import pandas as pd
import operator


def is_series(x):
    """
    returns true iff x is of type series
    """
    return type(x) == pd.Series


def div(x1, x2):
    x1, x2 = pd.Series(x1), pd.Series(x2)
    if all(np.abs(x2) > 0.0001):
        return x1/x2
    else:
        return pd.Series(0)


def exponent(x1):
    x1 = pd.Series(x1)
    if all(np.abs(x1) < 100):
        return np.exp(x1)
    return pd.Series(0)

def _logical(x1, x2, x3, x4):
    return np.where(x1 > x2, x3, x4)[0]

def power(x1, x2):
    x1, x2 = pd.Series(x1), pd.Series(x2)
    val = np.power(x1, x2)
    if not all(np.isfinite(val)):
        return pd.Series(0)
    else:
        return val

def square(x1):
    return power(x1, 2)

def sin(x):
    return np.sin(x)

def arcsin(x):
    return np.arcsin(x)

def cos(x):
    return np.cos(x)

def arccos(x):
    return np.arccos(x)

def tan(x):
    return np.tan(x)

def arctan(x):
    return np.arctan(x)

def log(x):
    x = pd.Series(x)
    if all(x > 0.000000001):
        return np.log(x)
    return pd.Series(0)


def abs(x):
    return np.abs(x)

def inv(x):
    x = pd.Series(x)
    if all(x > 0.0001):
        return 1/x
    return pd.Series(0)

def sqrt(x):
    x = pd.Series(x)
    if all(x > 0.000001):
        return np.sqrt(x)
    return 0


def add(x, y):
    #print(type(x))
    return operator.add(x, y)





func_dict = {
    "add" : add,
    "sub" : operator.sub,
    "mul" : operator.mul,
    "div" : div,
    "exp" : exponent,
    "pow" : power,
    "sin" : sin,
    "cos" : cos,
    "arcsin" : arcsin,
    "arccos" : arccos,
    "tan" : tan,
    "arctan" : arctan,
    "log" : log,
    "abs" : abs,
    "inv" : inv,
    "max" : max,
    "min" : min,
    "sqrt": sqrt,
    "square": square
    }