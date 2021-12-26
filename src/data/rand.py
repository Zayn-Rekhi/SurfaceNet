import numpy as np

SEED = 101 

def shuffle_array(*args):
    np.random.seed(seed=SEED)
    idxs = np.random.permutation(args[0].shape[0]) 
    return [arr[idxs] for arr in args]
