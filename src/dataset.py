import numpy as np
import pandas as pd
from scipy.io import loadmat
from h5py import File

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

def h5py_to_dataframe(h5py_object):
    key = [key for key in h5py_object if 'Dataset' in key][0]
    pos = pd.DataFrame(np.array(h5py_object[key]['Dpos']).T)
    neg = pd.DataFrame(np.array(h5py_object[key]['Dneg']).T)
    feat_cols = [f'Feat_{i+1:03}' for i in range(pos.shape[1])]
    pos.columns = feat_cols
    neg.columns = feat_cols
    pos['Target'] = 1
    neg['Target'] = 0
    data = pd.concat([pos, neg])
    metadata_cols = ['CauseGene', 'EffectGene', 'Replicate', 'Treatment', 'Pvalue']
    for c in metadata_cols:
        data[c] = np.nan
    cols = metadata_cols + feat_cols + ['Target']
    return data[cols]

def load_mat(mat_file, h5py=False):
    return h5py_to_dataframe(File(str(mat_file), 'r')) if h5py else \
           mat_to_dataframe(loadmat(str(mat_file)))
