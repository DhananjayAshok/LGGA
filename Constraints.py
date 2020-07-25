import numpy as np
import pandas as pd

def pairwise_symnetric_result(func, dls, X, y, col0="X0", col1="X1"):
    temp_clone = X.copy()
    temp_clone[col0] = X[col1]
    temp_clone[col1] = X[col0]
    predsym = dls.get_result(func, temp_clone, y)
    return predsym

def get_floored_max(series):
    return max(np.append(series, 0))

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

def snell_lgml_func(ind, dls=None, gen=None, threshold=2):
    if gen is None:
        return None, None
    X, y = gen()
    predr, predsym, i, r, n = snell_computations(ind, dls, X, y, threshold)
    nonzeros = np.abs(predsym) >= 0.001
    symnetry_violation = np.abs(predr - 1/predsym[nonzeros]) > threshold
    return get_union_slice([symnetry_violation], X, y)

def coloumb_computations(func, dls, X, y, threshold):
    preds = dls.get_result(func, X, y)
    predsym = pairwise_symnetric_result(func, dls, X, y)
    q1 = X['X0']
    q2 = X['X1']
    r = X['X2']
    q = y
    both_positive = (q1 > 0) & (q2 > 0)
    both_negative = (q1 < 0) & (q2 < 0)
    both_same_sign = both_negative | both_positive
    return preds, predsym, q1, q2, r, q, both_same_sign

def coloumb_constraints(dls, X, y, weight=10, threshold=2):
    func = dls.func
    preds, predsym, q1, q2, r, q, both_same_sign = coloumb_computations(func, dls, X, y, threshold)
    symntery_violation = np.abs(preds - predsym)
    same_sign_violation = -preds[both_same_sign] # If the results are positive then value will be neg if neg then value positive
    diff_sign_violation = preds[~both_same_sign]
    return weight * (get_floored_max(symntery_violation) + get_floored_max(same_sign_violation) + get_floored_max(diff_sign_violation))


def coloumb_lgml_func(ind, dls=None, gen=None, threshold=2):
    if gen is None:
        return None, None
    X, y = gen()
    preds, predsym, q1, q2, r, q, both_same_sign = coloumb_computations(ind, dls, X, y, threshold)
    symviolation = np.abs(preds - predsym) > threshold
    same_sign_violation = -preds[both_same_sign] > threshold
    diff_sign_violation = preds[~both_same_sign] > threshold
    return get_union_slice([symviolation, same_sign_violation, both_same_sign], X, y)

def reflection_computations(func, dls, X, y, threshold):
    preds = dls.get_result(func, X, y)
    predsym = pairwise_symnetric_result(func, dls, X, y)
    n1 = X['X0']
    n2 = X['X1']
    r = y
    return preds, predsym, n1, n2, r

def reflection_constraints(dls, X, y, weight=10, threshold=0.00001):
    func = dls.func
    predr, predsym, n1, n2, r = reflection_computations(func, dls, X, y, threshold)
    symnetry_violation = np.abs(predr - predsym)
    range_violation = np.abs(predr - 0.5) - 0.5
    return weight * (get_floored_max(symnetry_violation) + get_floored_max(range_violation))

def reflection_lgml_func(ind, dls=None, gen=None, threshold=0.00001):
    if gen is None:
        return None, None
    X, y = gen()
    predr, predsym, n1, n2, r = reflection_computations(ind, dls, X, y, threshold)
    symnetry_violation = np.abs(predr - predsym) > threshold
    range_violation = np.abs(predr - 0.5) > 0.5
    return get_union_slice([symnetry_violation, range_violation], X, y)
