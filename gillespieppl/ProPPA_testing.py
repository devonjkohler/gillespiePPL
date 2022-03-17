import sys
sys.path.append("../ProPPA")

import numpy as np
import scipy.stats as spst
import matplotlib.pyplot as plt
import pickle

import pandas as pd

lo_path = "data/Proppa_Models/LacOperon_one_var_infer.proppa"
lo_model = proppa.load_model(lo_path)

n_samples = 10000  # how many samples to take from the posterior

samples = lo_model.infer(n_samples)
samples = np.array(samples)