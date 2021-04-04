import tensorflow as tf
from numpy import prod as npprod

def rec(w):

    for value in w.values():
        if isinstance(value, dict):
            yield from rec(value)
        else:
            yield value

def get_weights(ws):
    # Parses weights out of weights dictionary
    weights = []
    for w in ws:
        if isinstance(w, dict):
            for w in list(rec(w)):
                weights.append(w)
        else:
            weights.append(w)

    return weights
