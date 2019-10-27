import functools
import numpy as np

def conjunction(*conditions):
    return functools.reduce(np.logical_and, conditions)

def union(*conditions):
    return functools.reduce(np.logical_or, conditions)