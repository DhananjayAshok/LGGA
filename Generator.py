import pandas as pd
import numpy as np
import os


def generate_pythogoras(no_samples=1000, input_range=(0, 500), save_path="data//"):
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
    df.to_csv(os.path.join(save_path, "pythogoras.csv"), index=False)