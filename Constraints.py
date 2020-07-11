import numpy as np
import pandas as pd

def pairwise_symnetric_result(func, dls, X, y, col0="X0", col1="X1"):
    temp_clone = X.copy()
    temp_clone[col0] = X[col1]
    temp_clone[col1] = X[col0]
    predsym = dls.get_result(func, temp_clone, y)
    return predsym

def get_union_slice(violations, X, y):
    overall = violations[0]
    for violation in violations[1:]:
        overall = (overall | violation)
    if not any(overall):
        return None, None
    else:
        return X[overall], y[overall]



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
    predsymr = pairwise_symnetric_result(func, dls, X, y)
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
    return get_union_slice([symviolation, x1violation, x2violation], X, y)
    


def snell_computations(func, dls, X, y, threshold):
    preds = dls.get_result(func, X, y)
    predsym = pairwise_symnetric_result(func, dls, X, y)
    i = X['X0']
    r = X['X1']
    n = y
    return preds, predsym, i, r, n

def snell_constraints(dls, X, y, weight=10, threshold=2):
    func = dls.func
    predr, predsym, i, r, n = snell_computations(func, dls, X, y, threshold)
    nonzeros = np.abs(predsym) >= 0.001
    symnetry_violation = np.abs(predr - 1/predsym[nonzeros])
    return weight * max(np.append(symnetry_violation, 0))

def snell_lgml_function(ind, dls=None, gen=None, threshold=2):
    if gen is None:
        return None, None
    X, y = gen()
    predr, predsym, i, r, n = snell_computations(func, dls, X, y, threshold)
    nonzeros = np.abs(predsym) >= 0.001
    symnetry_violation = np.abs(predr - 1/predsym[nonzeros]) > threshold
    return get_union_slice([sysymnetry_violation], X, y)

