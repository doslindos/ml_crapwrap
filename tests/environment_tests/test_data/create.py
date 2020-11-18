import numpy as np

def create_numpy_data(dimensions, content, datatype):
    return np.full(dimensions, content, dtype=datatype)
