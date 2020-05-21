import numpy as np



def exponent(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 100, np.exp(x1), 0.)


def powere(x1, x2):
    with np.errstate(over="ignore"):
        return np.where(np.abs(x2) < 100 and np.abs(x1) > 0.001, math.pow(x1, x2), 0)

def _logical(x1, x2, x3, x4):
    return np.where(x1 > x2, x3, x4)

def power(x1, x2):
    val = np.power(x1, x2)
    if not np.isfinite(val)[0]:
        return 0
    else:
        return val