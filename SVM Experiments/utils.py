import re
import numpy as np
import pandas as pd
from scipy.io import loadmat

def mat_to_dataframe(mat_object):
    key = [key for key in mat_object if 'Dataset' in key][0]
    columns = mat_object[key][0,0][4][0].strip().split(';')
    df = pd.DataFrame()
    for i in range(5):
        df[columns[i]] = np.concatenate([
                        mat_object[key][0,0][2][0][i].reshape(-1),
                        mat_object[key][0,0][3][0][i].reshape(-1)
        ])
    for i in range(18):
        df[columns[i+5]] = np.concatenate([
                        mat_object[key][0,0][0][:,i],
                        mat_object[key][0,0][1][:,i]
        ])
    for i in [0, 1, 3]:
        df[columns[i]] = df[columns[i]].apply(lambda x: x[0])
    df['Target'] = [0]*len(mat_object[key][0,0][0]) + \
                   [1]*len(mat_object[key][0,0][1])
    return df

def load_mat(mat_file):
    return mat_to_dataframe(loadmat(mat_file))