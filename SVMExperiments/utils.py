import re
import numpy as np
import pandas as pd
from scipy.io import loadmat

# 0: Positive class data (16662, 18)
# 1: Negative class data (673200, 18)
# 2: Positive class additional info [Cause Gene, Effect Gene, Replicate, Treatment, Pvalue]
# 3: Negative class additional info [Cause Gene, Effect Gene, Replicate, Treatment, Pvalue]
# 4: Column names (separated by ;)
# 5: [[6]]

def mat_to_dataframe(mat_object):
    key = [key for key in mat_object if 'Dataset' in key][0]
    columns = mat_object[key][0,0][4][0].strip().split(';')
    df = pd.DataFrame()
    # Get CauseGene, EffectGene, Replicate, Treatment, Pvalue
    for i in range(5):
        df[columns[i]] = np.concatenate([
                        mat_object[key][0,0][2][0][i].reshape(-1),
                        mat_object[key][0,0][3][0][i].reshape(-1)
        ])
    # Get expression data
    for i in range(mat_object[key][0,0][0].shape[1]):
        df[columns[i+5]] = np.concatenate([
                        mat_object[key][0,0][0][:,i],
                        mat_object[key][0,0][1][:,i]
        ])
    # Convert [str] to str
    for i in [0, 1, 3]:
        df[columns[i]] = df[columns[i]].apply(lambda x: x[0])
    # Make binary target
    df['Target'] = [1]*len(mat_object[key][0,0][0]) + \
                   [0]*len(mat_object[key][0,0][1])
    return df

def load_mat(mat_file):
    return mat_to_dataframe(loadmat(mat_file))