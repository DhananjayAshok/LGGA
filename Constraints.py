import numpy as np

def triangle_rule(dls, X, y, weight=1000):
    func = dls.func
    c = dls.get_result(func, X, y)
    a = X['X0']
    b = X['X1']
    flags = a + b < c
    nviolations = np.mean(flags)*100
    return weight*nviolations


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


