import re
import multiprocessing as mp
from functools import partial
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix

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
    return mat_to_dataframe(loadmat(str(mat_file)))

def log_metrics(targets, preds, run, feat_ranking, classifier, n_features, description):
    (tn, fp), (fn, tp) = confusion_matrix(targets, preds)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    gmean = np.sqrt(tp / (tp + fn) * tn / (tn + fp))                              
    log = f'Run: {run} | {feat_ranking} | {classifier} | NumFeatures: {n_features} | ' \
          f'{description} | TPR: {tpr:.4f} | TNR: {tnr:.4f} | GMean: {gmean:.4f} | ' \
          f'TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}'
    return tpr, tnr, gmean, log

def bag_idx(n_samples, x):
    return range(n_samples * (x-1), n_samples * x)

def get_bag_data(data, bags, n_samples, bag_id, features):
    '''
    data: pd.DataFrame
    bags: numpy matrix
    bag_idx: bag selector function
    bag_id: int
    features: list of ints
    '''
    return data[[x for i in bags[:, bag_id] \
                    for x in bag_idx(n_samples, i)]][:, features]

def predict_bag(bag_id, trn_neg, trn_pos, test_data, test_labels,
                 run, classifier, classifier_name, params, 
                 feature_ranker, features, bags, n_samples):
    trn_neg_bag_data = get_bag_data(trn_neg, bags['NegGenePairsBag'], 
                                    n_samples, bag_id, features)
    trn_pos_bag_data = get_bag_data(trn_pos, bags['PosGenePairsBag'], 
                                    n_samples, bag_id, features)

    train_data = np.concatenate([trn_pos_bag_data, trn_neg_bag_data])
    train_labels = np.concatenate([np.ones(len(trn_pos_bag_data)), 
                                   np.zeros(len(trn_neg_bag_data))])

    model = classifier(**params)
    model.fit(train_data, train_labels)
    test_preds = model.predict(test_data)

    tpr, tnr, gmean, log = log_metrics(test_labels, test_preds, 
                                   run, feature_ranker, classifier_name,
                                   len(features), f'Bag: {bag_id}')

    return [tpr, tnr, gmean], log, test_preds

import ctypes

def evaluate_bags(trn_neg, trn_pos, test_data, test_labels, 
                  run, classifier, classifier_name, params, 
                  feature_ranker, features, bags, n_samples, num_bags, f,
                  idx_classifier=None, n_classifiers=None, idx_featrank=None, n_featrank=None,
                  idx_topfeat=None, n_topfeats=None):
    # all columns except metadata (5:) and target (:-1)    
    trn_neg = trn_neg.iloc[:, 5:-1].values
    trn_pos = trn_pos.iloc[:, 5:-1].values
    test_data = test_data.values
    test_labels = test_labels.values

    # trn_neg_shape = trn_neg.shape
    # trn_neg = mp.Array('d', trn_neg.values.reshape(-1))
    # trn_neg = np.ctypeslib.as_array(trn_neg.get_obj())
    # trn_neg = trn_neg.reshape(trn_neg_shape)
    
    # trn_pos_shape = trn_pos.shape
    # trn_pos = mp.Array('d', trn_pos.values.reshape(-1))
    # trn_pos = np.ctypeslib.as_array(trn_pos.get_obj())
    # trn_pos = trn_pos.reshape(trn_pos_shape)
    
    # test_data_shape = test_data.shape
    # test_data = mp.Array('d', test_data.values.reshape(-1))
    # test_data = np.ctypeslib.as_array(test_data.get_obj())
    # test_data = test_data.reshape(test_data_shape)
    
    # test_labels_shape = test_labels.shape
    # test_labels = mp.Array('d', test_labels.values.reshape(-1))
    # test_labels = np.ctypeslib.as_array(test_labels.get_obj())
    # test_labels = test_labels.reshape(test_labels_shape)

    partial_predict = partial(predict_bag, trn_neg=trn_neg, trn_pos=trn_pos, 
                            test_data=test_data, test_labels=test_labels, run=run,
                            classifier=classifier, classifier_name=classifier_name,
                            params=params, feature_ranker=feature_ranker, features=features,
                            bags=bags, n_samples=n_samples)

    # pool = mp.Pool()
    # results = pool.map(partial_predict, range(max(num_bags)))
    # pool.close()
    # pool.join()
    
    if n_classifiers is not None:
        n_bags = max(num_bags)
        total = n_bags * n_topfeats * n_featrank * n_classifiers
    results = []
    for bag_id in range(max(num_bags)):
        results.append(partial_predict(bag_id))
        
        progress_str = ''
        if n_classifiers is not None:
            # print(f'bag_id: {bag_id} | n_bags: {n_bags} | idx_featrank: {idx_featrank} | '
            #       f'n_featrank: {n_featrank} | idx_classifier: {idx_classifier} | n_classifiers: {n_classifiers}')
            progress = bag_id + idx_topfeat * n_bags + \
                       idx_featrank * n_topfeats * n_bags + \
                       idx_classifier * n_featrank * n_topfeats * n_bags
            progress_str = f'{progress/total*100:.2f}% | '
        
        print(progress_str + results[-1][1], file=f, flush=True)
    
    metrics, logs, preds = zip(*results)
    # for log in logs:
    #     print(log)

    preds = np.array(preds)
    metrics = np.array(metrics)
    for n_bags in num_bags:
        # [100, 200, 300]
        log = f'Mean TPR: {metrics[:n_bags, 0].mean():.4f} | ' \
              f'Mean TNR: {metrics[:n_bags, 1].mean():.4f} | ' \
              f'Mean GMean: {metrics[:n_bags, 2].mean():.4f}'
        print(log, file=f, flush=True)
        tpr, tnr, gmean, log = log_metrics(test_labels, 
                    (preds[:n_bags, :].mean(axis=0) > 0.5).astype(int), 
                    run, feature_ranker, classifier_name, 
                    len(features), f'Ensemble of {n_bags} bags')
        print(log, file=f, flush=True)

    # print()       
    
    
import sys

def classify_feature_rank(DataCV_dir, Bags_dir, FeatRanking_dir, classifiers, params, 
                          dataset, feature_set, treatment, 
                          pval_pos_threshold, num_bags, num_runs,
                          num_top_features, test_all_features=False):
    
    file_name = f'{feature_set} - {dataset} - {treatment}.txt'
    
    with open(file_name, 'w') as f:
    
        f = sys.stdout
        print(f'Feature set: {feature_set} | Dataset: {dataset} | Treatment: {treatment} | '
              f'Pval: {pval_pos_threshold}', file=f, flush=True)
        print(f'Num bags: {num_bags} | Num runs: {num_runs} | Num features: {num_top_features} | All features: {test_all_features}', 
              file=f, flush=True)

        train = load_mat(DataCV_dir/f'{feature_set}/{dataset}_{treatment}'
                                    f'({pval_pos_threshold}).trn.mat')
        test = load_mat(DataCV_dir/f'{feature_set}/{dataset}_{treatment}'
                                   f'({pval_pos_threshold}).tst.mat')
        feat_rank = pd.read_csv(FeatRanking_dir/f'{feature_set}/{dataset}_{treatment}'
                                                f'({pval_pos_threshold}).csv', sep=';')
        trn_neg = train[train.Target == 0].reset_index(drop=True)
        trn_pos = train[train.Target == 1].reset_index(drop=True)
        
        for run in num_runs:
            # [1, 2, 3, 4, 5]
            # print(f'Run: {run}') 
            bags = loadmat(Bags_dir/f'{dataset}({pval_pos_threshold})_Bags{run}.mat')
            # Sanity check
            n_samples = 6 if treatment == 'All' else 3
            assert bags['NegGenePairsBag'].max() == len(train[train.Target==0]) // n_samples
            assert bags['PosGenePairsBag'].max() == len(train[train.Target==1]) // n_samples

            feat_rank_rows = feat_rank[feat_rank.Run == run]

            n_classifiers = len(classifiers)
            for idx_classifier, (classifier, param) in enumerate(zip(classifiers, params)):
                # [LGBMClassifier, LinearSVC, RandomForestClassifier]
                classifier_name = str(classifier).strip("<>'").split('.')[-1]
                # print(' '*3, f'Classifier: {classifier_name}')

                n_featranks = len(feat_rank_rows)
                for idx_featrank, (_, feat_rank_row) in enumerate(feat_rank_rows.iterrows()):
                    # [Entropy, Ttest, Brattacharyya, Wilcoxon]
                    # print(' '*6, f'Feature ranking method: {feat_rank_row.Method}')             

                    n_topfeats = len(num_top_features)
                    for idx_topfeat, num_top_feats in enumerate(num_top_features):
                        # [2, 4, 6, 7, 10, 12] matlab index starts at 1
                        features = [col - 1 for col in feat_rank_row[2:2+num_top_feats]]
                        # print(' '*9, f'Features: {features}')

                        test_data = test.iloc[:,5:-1].iloc[:,features]
                        test_labels = test.Target
                        evaluate_bags(trn_neg, trn_pos, test_data, test_labels, 
                                      run, classifier, classifier_name, param, 
                                      feat_rank_row.Method, features, bags, n_samples, num_bags, f,
                                      idx_classifier, n_classifiers, idx_featrank, n_featranks,
                                      idx_topfeat, n_topfeats)    
                if test_all_features:
                    # print(' '*6, f'All features')
                    # use all features (6 metadata columns)
                    num_top_feats = train.shape[1] - 6
                    # [2, 4, 6, 7, 10, 12] matlab index starts at 1
                    features = list(range(num_top_feats))
                    # print(' '*9, f'Features: {features}')

                    test_data = test.iloc[:,5:-1].iloc[:,features]
                    test_labels = test.Target
                    evaluate_bags(trn_neg, trn_pos, test_data, test_labels, 
                                  run, classifier, classifier_name, param, 
                                  feat_rank_row.Method, features, bags, n_samples, num_bags, f)
                    
                    
                          