import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import loadmat
import scipy


d = loadmat("mhad_data/sub1/front/catch/a_s_t1_skel_K2.mat", simplify_cells = True)

d.keys()
d
d['S_K2'][0,0]


d['depth_K2'].shape
d['d_iner'].shape