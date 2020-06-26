import numpy as np

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


def resistance_constraints(dls, X, y, weight=1000, threshold=2):
    func = dls.func
    predr = dls.get_result(func, X, y)
    temp_clone = X.copy()
    temp_clone['X1'] = X['X0']
    temp_clone['X0'] = X['X1']
    predsymr = dls.get_result(func, temp_clone, y)
    r1 = X['X0']
    r2 = X['X1']
    r = y
    symmnetry_violation = np.abs(predr - predsymr)
    x1specific = r1 - predr
    x2specific = r2 - predr
    rviolations = max(np.append(symmnetry_violation,0)) + max(np.append(x1specific, 0)) + max(np.append(x2specific, 0))
    return weight * rviolations



