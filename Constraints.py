import numpy as np
import pandas as pd

def triangle_rule(dls, X, y, weight=1000):
    func = dls.func
    c = dls.get_result(func, X, y)
    a = X['X0']
    b = X['X1']
    diff = c - b - a
    #flags = a + b < c
    #nviolations = np.mean(flags)*100
    #return weight*nviolations
    return max(np.append(diff, 0))*weight


def semiperimeter_rule(dls, X, y, weight=1000, threshold=2):
    func = dls.func
    predc = dls.get_result(func, X, y)
    a = X['X0']
    b = X['X1']
    c = y
    s = (a + b + c)/2
    flags = np.abs((s-a)*(s-b) - s*(s-predc)) > threshold 
    rviolations = np.mean(flags)*100
    return weight*rviolations


def resistance_computations(func, dls, X, y, threshold):
    predr = dls.get_result(func, X, y)
    temp_clone = X.copy()
    temp_clone['X1'] = X['X0']
    temp_clone['X0'] = X['X1']
    predsymr = dls.get_result(func, temp_clone, y)
    r1 = X['X0']
    r2 = X['X1']
    r = y
    return predr, predsymr, r1, r2, r

def resistance_constraints(dls, X, y, weight=1000, threshold=2):
    func = dls.func
    predr, predsymr, r1, r2, r = resistance_computations(func,dls, X, y, threshold)
    symmnetry_violation = np.abs(predr - predsymr)
    x1specific = predr - r1
    x2specific = predr - r2
    rviolations = max(np.append(symmnetry_violation,0)) + max(np.append(x1specific, 0)) + max(np.append(x2specific, 0))
    return weight * rviolations


def resistance_lgml_func(ind, dls=None, gen=None, threshold=2):
    if gen is None:
        return None, None
    X, y = gen()
    predr, predsymr, r1, r2, r = resistance_computations(ind,dls, X, y, threshold)
    symviolation = np.abs(predr - predsymr) > threshold
    x1violation = r1 < predr
    x2violation = r2 < predr
    overall = (symviolation | x1violation) | (x2violation)
    if not any(overall):
        return None, None
    else:
        return X[overall], y[overall]


