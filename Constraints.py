import numpy as np
import pandas as pd

def pairwise_symnetric_result(func, dls, X, y, col0="X0", col1="X1"):
    temp_clone = X.copy()
    temp_clone[col0] = X[col1]
    temp_clone[col1] = X[col0]
    predsym = dls.get_result(func, temp_clone, y)
    return temp_clone, predsym

def zero_result(func, dls, X, y, cols=["X0"]):
    temp_clone = X.copy()
    for col in cols:
        temp_clone[col] = 0
    return temp_clone, dls.get_result(func, temp_clone, y)

def equality_result(func, dls, X, y, cols=["X0", "X1"]):
    temp_clone = X.copy()
    for col in cols[1:]:
        temp_clone[col] = X[cols[0]]
    predeq = dls.get_result(func, temp_clone, y)
    return temp_clone, predeq


def get_floored_max(series, floor=0):
    return max(np.append(series, floor))

def get_union_slice(violations, X, y):
    overall = violations[0]
    for violation in violations[1:]:
        overall = (overall | violation)
    if not any(overall):
        return None, None
    else:
        return X[overall], y[overall]


def get_union_slice(violations):
    """
    Expects violation to be [(boolean frame, X, y)....]
    """
    Xs = pd.concat([violation[1][violation[0]] for violation in violations])
    ys = pd.concat([violation[2][violation[0]] for violation in violations])
    print(f"Adding {len(ys)} points")
    return Xs, ys
    

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Constraints and LGML functions declared Below
    
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def resistance_computations(func, dls, X, y, threshold):
    predr = dls.get_result(func, X, y)
    Xsym, predsymr = pairwise_symnetric_result(func, dls, X, y)
    X0_zero, predr1zero = zero_result(func, dls, X, y, cols=["X0"])
    X1_zero, predr2zero = zero_result(func, dls, X, y, cols=["X1"])
    X_eq, predeq = equality_result(func, dls, X, y, cols=["X0", "X1"])
    r1 = X['X0']
    r2 = X['X1']
    r = y
    return Xsym, X0_zero, X1_zero, X_eq, predr, predsymr, predr1zero, predr2zero, predeq, r1, r2, r

def resistance_constraints(dls, X, y, weight=1000, threshold=2):
    func = dls.func
    Xsym, X0_zero, X1_zero, X_eq, predr, predsymr, predr1zero, predr2zero, predeq, r1, r2, r = resistance_computations(func,dls, X, y, threshold)
    symmnetry_violation = np.abs(predr - predsymr)
    x1specific = predr - r1
    x2specific = predr - r2
    
    rviolations = max(np.append(symmnetry_violation,0)) + max(np.append(x1specific, 0)) + max(np.append(x2specific, 0))
    return weight * rviolations


def resistance_lgml_func(ind, X=None, y=None, dls=None, threshold=2):
    if X is None or y is None:
        return None, None
    Xsym, X0_zero, X1_zero, X_eq, predr, predsymr, predr1zero, predr2zero, predeq, r1, r2, r = resistance_computations(ind, dls, X, y, threshold)
    symviolation = np.abs(predr - predsymr) > threshold
    x1violation = r1 < predr
    x2violation = r2 < predr
    r1zeroviolation = np.abs(predr1zero) > threshold
    r2zeroviolation = np.abs(predr2zero) > threshold
    equality_violation = np.abs(r1/2 - predeq) > threshold # Equality result function makes all equal to first input column
    zero_col = pd.Series(np.zeros(shape=predr.shape))
    return get_union_slice([(symviolation, Xsym, y), (x1violation, X, y), (x2violation, X, y), (r1zeroviolation, X0_zero, zero_col), (r1zeroviolation, X1_zero, zero_col), (equality_violation, X_eq, r1/2)])
    


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


def gas_computations(func, dls, X, y, threshold):
    p = X['X0']
    v = X['X1']
    n = X['X2']
    t = X['X3']
    r = y
    pred = dls.get_result(func, X, y)
    predsympv = pairwise_symnetric_result(func, dls, X, y)
    predsymnt = pairwise_symnetric_result(func, dls, X, y, col0="X2", col1="X3")
    predzerop = zero_result(func, dls, X, y)
    predzerov = zero_result(func, dls, X, y, cols=["X1"])

    return p, v, n, t, r, pred, predsympv, predsymnt, predzerop, predzerov

def gas_constraints(dls, X, y, weight=10, threshold=0.00001):
    func = dls.func
    p, v, n, t, r, pred, predsympv, predsymnt, predzerop, predzerov = gas_computations(func, dls, X, y, threshold)
    symnetry_frames = [predsympv, predsymnt]
    symnetry_violations = [np.abs(pred - predsym) for predsym in symnetry_frames]
    zero_frames = [predzerop, predzerov]
    zero_violations = [zero_frame - 0.001 for zero_frame in zero_frames]
    return weight * (sum([get_floored_max(violation) for violation in symnetry_violations]) + sum([get_floored_max(violation) for violation in zero_violations]))

def gas_lgml_func(ind, dls=None, gen=None, threshold=0.00001):
    if gen is None:
        return None, None
    X, y = gen()
    p, v, n, t, r, pred, predsympv, predsymnt, predzerop, predzerov = gas_computations(ind, dls, X, y, threshold)
    symnetry_frames = [predsympv, predsymnt]
    zero_frames = [predzerop, predzerov]
    symnetry_violations = [np.abs(pred - predsym) > threshold for predsym in symnetry_frames]
    zero_violations = [np.abs(frame) > threshold for frame in zero_frames]
    return get_union_slice(symnetry_violations+zero_violations, X, y)


def distance_computations(func, dls, X, y, threshold):
    x0 = X['X0']
    x1 = X['X1']
    y0 = X['X2']
    y1 = X['X3']
    d = y
    pred = dls.get_result(func, X, y)
    pred_symx = pairwise_symnetric_result(func, dls, X, y)
    pred_symy = pairwise_symnetric_result(func, dls, X, y, col0="X2", col1="X3")
    pred_zero_x0 = zero_result(func, dls, X, y, cols=["X1", "X2", "X3"])
    pred_zero_x1 = zero_result(func, dls, X, y, cols=["X0", "X2", "X3"])
    pred_zero_y0 = zero_result(func, dls, X, y, cols=["X0", "X1", "X3"])
    pred_zero_y1 = zero_result(func, dls, X, y, cols=["X0", "X1", "X2"])
    pred_eq = equality_result(func, dls, X, y, cols=["X0", "X1", "X2", "X3"])
    return x0, x1, y0, y1, d, pred, pred_symx, pred_symy, pred_zero_x0, pred_zero_x1, pred_zero_y0, pred_zero_y1, pred_eq


def distance_constraints(dls, X, y, weight=10, threshold=0.00001):
    func = dls.func
    x0, x1, y0, y1, d, pred, pred_symx, pred_symy, pred_zero_x0, pred_zero_x1, pred_zero_y0, pred_zero_y1, pred_eq = distance_computations(func, dls, X, y, threshold)
    symnetry_frames = [pred_symx, pred_symy]
    symnetry_violations = [np.abs(pred - predsym) for predsym in symnetry_frames]
    x0_violation = np.abs(x0 - pred_zero_x0)
    x1_violation = np.abs(x1 - pred_zero_x1)
    y0_violation = np.abs(y0 - pred_zero_y0)
    y1_violation = np.abs(y1 - pred_zero_y1)
    value_violations = [x0_violation, x1_violation, y0_violation, y1_violation]
    equality_violation = pred_eq - threshold
    return weight * (sum([get_floored_max(violation) for violation in symnetry_violations]) + sum([get_floored_max(violation) for violation in value_violations]) + get_floored_max(equality_violation))

def distance_lgml_func(ind, dls=None, gen=None, threshold=0.00001):
    if gen is None:
        return None, None
    X, y = gen()
    x0, x1, y0, y1, d, pred, pred_symx, pred_symy, pred_zero_x0, pred_zero_x1, pred_zero_y0, pred_zero_y1, pred_eq = distance_computations(ind, dls, X, y, threshold)
    symnetry_frames = [pred_symx, pred_symy]
    symnetry_violations = [np.abs(pred - predsym) > threshold for predsym in symnetry_frames]
    x0_violation = np.abs(x0 - pred_zero_x0) > threshold
    x1_violation = np.abs(x1 - pred_zero_x1) > threshold
    y0_violation = np.abs(y0 - pred_zero_y0) > threshold
    y1_violation = np.abs(y1 - pred_zero_y1) > threshold
    value_violations = [x0_violation, x1_violation, y0_violation, y1_violation]
    equality_violation = pred_eq > threshold
    violations = symnetry_violations+value_violations+[equality_violation]
    return get_union_slice(violations, X, y)


def normal_computations(func, dls, X, y, threshold):
    x = X['X0']
    n = y
    pred = dls.get_result(func, X, y)
    temp_x = X.copy()
    temp_x['X0'] = -x
    predneg = dls.get_result(func, temp_x, y)
    predzero = zero_result(func, dls, X, y, cols=["X0"])
    max_pred = max(pred)
    return x, n, pred, predneg, predzero, max_pred

def normal_constraints(dls, X, y, weight=10, threshold=0.0001):
    func = dls.func
    x, n, pred, predneg, predzero, max_pred = normal_computations(func,dls, X,y, threshold)
    negviolation = np.abs(pred - predzero)
    zero_violation = np.abs(predzero - 0.1591549)
    max_violation = max_pred - predzero
    min_violation = -pred
    return weight * (get_floored_max(negviolation) + get_floored_max(zero_violation) + get_floored_max(max_violation) + get_floored_max(min_violation))


def normal_lgml_func(ind, dls, X, y, threshold=0.00001):
    if gen is None:
        return None, None
    X, y = gen()
    x, n, pred, predneg, predzero, max_pred = normal_computations(ind,dls, X,y, threshold)
    negviolation = np.abs(pred - predzero) > threshold
    zero_violation = np.abs(predzero - 0.1591549) > threshold
    max_violation = max_pred - predzero > threshold
    min_violation = -pred > threshold
    return get_union_slice([negviolation, zero_violation, max_violation,  min_violation], X, y)


