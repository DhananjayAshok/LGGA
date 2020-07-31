"""
Process to add another equation:
1. Add generator
2. Run generator to produce X and y for testing
3. Add generator to generator_dict
4. Add constraints
5. Add the equation to OtherEquations.csv
"""

import pandas as pd
import numpy as np
import os
from DataExtraction import create_dataset






def get_generator_generic(equation_id, no_samples=1000, input_range=(-100, 100), noise_range=(0,0), master_file="FeynmanEquations.csv"):
    """
    Returns a function gen such that calls to gen return an unpacked tuple X, y for given parameters with no saving or loading

    gen will take in an optional parameter no_samples
    """
    def gen(no_samples=no_samples):
        df = create_dataset(equation_id, no_samples=no_samples, input_range=input_range,save=False, load=False, noise_range=noise_range, master_file=master_file)
        return df.drop("target", axis=1), df['target']
    return gen


def get_generator_pythogoras(no_samples=1000, input_range=(0, 500), to_save=False, save_path="data//"):
    """
    Returns a function gen such that calls to gen return X, y as per pythogoras requirements with saving optional

    gen will take in an optional parameter no_samples
    """
    def gen(no_samples=no_samples):
        inputs = []
        outputs = []

        while len(outputs) < no_samples:
            a = np.random.default_rng().uniform(input_range[0] ,input_range[1])
            b = np.random.default_rng().uniform(input_range[0] ,input_range[1])
            if a == 0 or b == 0:
                continue
            c = np.sqrt(a**2 + b**2)
            if a + b < c:
                continue
            else:
                inputs.append([a, b])
                outputs.append(c)
        c = np.array(outputs)
        df = pd.DataFrame(inputs, columns=["X0", "X1"])
        df['target'] = c
        if to_save:
            df.to_csv(os.path.join(save_path, "pythogoras.csv"), index=False)
        return df.drop('target', axis=1), df['target']
    return gen

def get_generator_resistance(no_samples=1000, input_range=(0, 500), to_save=False, save_path="data//"):
    """
    Returns a function gen such that calls to gen return X, y as per resistance requirements with saving optional

    gen will take in an optional parameter no_samples
    """
    def gen(no_samples=no_samples):
        inputs = []
        outputs = []

        while len(outputs) < no_samples:
            r1 = np.random.default_rng().uniform(input_range[0] ,input_range[1])
            r2 = np.random.default_rng().uniform(input_range[0] ,input_range[1])
            if r1 <= 0 or r2 <= 0 or r1+r2 <=0:
                continue
            r = (r1*r2)/(r1+r2)
            inputs.append([r1, r2])
            outputs.append(r)
        r = np.array(outputs)
        df = pd.DataFrame(inputs, columns=["X0", "X1"])
        df['target'] = r
        if to_save:
            df.to_csv(os.path.join(save_path, "resistance.csv"), index=False)
        return df.drop('target', axis=1), df['target']
    return gen

def get_generator_snell(no_samples=1000, input_range=(0, 1.5708), to_save=False, save_path="data//"):
    """
    Returns a function gen such that calls to gen return X, y as per snell requirements with saving optional

    gen will take in an optional parameter no_samples
    """
    def gen(no_samples=no_samples):
        inputs = []
        outputs = []
        index = 0
        while index < no_samples:
            i = np.random.default_rng().uniform(input_range[0] ,input_range[1])
            r = np.random.default_rng().uniform(input_range[0] ,input_range[1])
            if -0.0001 <= np.sin(i) <= 0.0001 or -0.0001 <= np.sin(r) <= 0.0001:
                continue
            n = np.sin(i)/np.sin(r)
            inputs.append([i, r])
            outputs.append(n)
            index+=1
        n = np.array(outputs)
        df = pd.DataFrame(inputs, columns=["X0", "X1"])
        df['target'] = n
        if to_save:
            df.to_csv(os.path.join(save_path, "snell.csv"), index=False)
        return df.drop('target', axis=1), df['target']
    return gen

def get_generator_coloumb(no_samples=1000, input_range=(-100, 100), to_save=False, save_path="data//"):
    """
    Returns a function gen such that calls to gen return X, y as per coloumb requirements with saving optional

    gen will take in an optional parameter no_samples
    """
    def gen(no_samples=no_samples):
        inputs = []
        outputs = []
        index = 0
        while index < no_samples:

            q1 = np.random.default_rng().uniform(input_range[0] ,input_range[1])
            q2 = np.random.default_rng().uniform(input_range[0] ,input_range[1])
            r = np.random.default_rng().uniform(input_range[0] , input_range[1]**0.5)            
            if r <= 0.0001:
                continue
            q = q1*q2/(r**2)
            inputs.append([q1, q2, r])
            outputs.append(q)
            index+=1
        q = np.array(outputs)
        df = pd.DataFrame(inputs, columns=["X0", "X1", "X2"])
        df['target'] = q
        if to_save:
            df.to_csv(os.path.join(save_path, "coloumb.csv"), index=False)
        return df.drop('target', axis=1), df['target']
    return gen


def get_generator_reflection(no_samples=1000, input_range=(0, 100), to_save=False, save_path="data//"):
    """
    Returns a function gen such that calls to gen return X, y as per reflection requirements with saving optional

    gen will take in an optional parameter no_samples
    """
    def gen(no_samples=no_samples):
        inputs = []
        outputs = []
        index = 0
        while index < no_samples:
            n1 = np.random.default_rng().uniform(input_range[0] ,input_range[1])
            n2 = np.random.default_rng().uniform(input_range[0] ,input_range[1])
            if n1+n2 <= 0.0001:
                continue
            r = np.abs((n1-n2)/(n1+n2))**2
            inputs.append([n1,n2])
            outputs.append(r)
            index+=1
        r = np.array(outputs)
        df = pd.DataFrame(inputs, columns=["X0", "X1"])
        df['target'] = r
        if to_save:
            df.to_csv(os.path.join(save_path, "reflection.csv"), index=False)
        return df.drop('target', axis=1), df['target']
    return gen

def get_generator_gas(no_samples=1000, input_range=(0, 100), to_save=False, save_path="data//"):
    """
    Returns a function gen such that calls to gen return X, y as per gas constant requirements with saving optional

    gen will take in an optional parameter no_samples
    """
    def gen(no_samples=no_samples):
        inputs = []
        outputs = []
        index = 0
        while index < no_samples:
            p = np.random.default_rng().uniform(input_range[0] ,input_range[1])
            v = np.random.default_rng().uniform(input_range[0] ,input_range[1])
            n = np.random.default_rng().uniform(input_range[0]+0.0001 ,np.sqrt(input_range[1]))
            t = np.random.default_rng().uniform(input_range[0]+0.0001 ,np.sqrt(input_range[1]))
            if t*n <= 0.0001:
                continue
            r = (p*v)/(n*t)
            inputs.append([p, v, n, t])
            outputs.append(r)
            index+=1
        r = np.array(outputs)
        df = pd.DataFrame(inputs, columns=["X0", "X1", "X2", "X3"])
        df['target'] = r
        if to_save:
            df.to_csv(os.path.join(save_path, "gas.csv"), index=False)
        return df.drop('target', axis=1), df['target']
    return gen

def get_generator_distance(no_samples=1000, input_range=(-100, 100), to_save=False, save_path="data//"):
    """
    Returns a function gen such that calls to gen return X, y as per distance requirements with saving optional

    gen will take in an optional parameter no_samples
    """
    def gen(no_samples=no_samples):
        inputs = []
        outputs = []
        index = 0
        while index < no_samples:
            x0 = np.random.default_rng().uniform(input_range[0] ,input_range[1])
            x1 = np.random.default_rng().uniform(input_range[0] ,input_range[1])
            y0 = np.random.default_rng().uniform(input_range[0] ,input_range[1])
            y1 = np.random.default_rng().uniform(input_range[0] ,input_range[1])
            
            d = np.sqrt((x1-x0)**2 + (y1-y0)**2)
            inputs.append([x0, x1, y0, y1])
            outputs.append(d)
            index+=1
        d = np.array(outputs)
        df = pd.DataFrame(inputs, columns=["X0", "X1", "X2", "X3"])
        df['target'] = d
        if to_save:
            df.to_csv(os.path.join(save_path, "distance.csv"), index=False)
        return df.drop('target', axis=1), df['target']
    return gen


generator_dict = {
    "pythogoras": get_generator_pythogoras,
    "resistance": get_generator_resistance,
    "snell": get_generator_snell,
    "coloumb": get_generator_coloumb,
    "reflection": get_generator_reflection,
    "gas": get_generator_gas,
    "distance": get_generator_distance
    }