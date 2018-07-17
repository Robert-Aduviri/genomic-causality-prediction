import re
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

def get_bag_data(data, bags, bag_idx, bag_id, features):
    '''
    data: pd.DataFrame
    bags: numpy matrix
    bag_idx: bag selector function
    bag_id: int
    features: list of ints
    '''
    bag = data.loc[[x for i in bags[:, bag_id] \
                    for x in bag_idx(i)]]
    # all columns except metadata (5:) and target (:-1)
    return bag.iloc[:, 5:-1].iloc[:, features]

def predict_bags(trn_neg, trn_pos, test_data, test_labels,
                 run, classifier, classifier_name, param, 
                 feature_ranker, features, bags, bag_idx, bag_id):
    trn_neg_bag_data = get_bag_data(trn_neg, bags['NegGenePairsBag'], 
                                    bag_idx, bag_id, features)
    trn_pos_bag_data = get_bag_data(trn_pos, bags['PosGenePairsBag'], 
                                    bag_idx, bag_id, features)

    train_data = np.concatenate([trn_pos_bag_data, trn_neg_bag_data])
    train_labels = np.concatenate([np.ones(len(trn_pos_bag_data)), 
                                   np.zeros(len(trn_neg_bag_data))])

    model = classifier()
    model.fit(train_data, train_labels)
    test_preds = model.predict(test_data)
    all_preds.append(test_preds)
    tpr, tnr, gmean, log = log_metrics(test_labels, test_preds, 
                                   run, feature_ranker, classifier_name,
                                   len(features), f'Bag: {bag_id}')
    return tpr, tnr, gmean, log

def evaluate_bags(trn_neg, trn_pos, test_data, test_labels, 
                  run, classifier, classifier_name, param, 
                  feature_ranker, features, bags, bag_idx, num_bags):
    
    results = []
    for bag_id in range(max(num_bags)):
        results.append(predict_bags(trn_neg, trn_pos, test_data, test_labels,
                 run, classifier, classifier_name, param, 
                 feature_ranker, features, bags, bag_idx, bag_id))

    for log in all_logs:
        print(log)

    all_preds = np.array(all_preds)
    all_metrics = np.array(all_metrics)
    for n_bags in num_bags:
        # [100, 200, 300]
        print(' '*12, f'Number of bags: {n_bags}')
        log = f'Mean TPR: {all_metrics[:n_bags, 0].mean():.4f} | ' \
              f'Mean TNR: {all_metrics[:n_bags, 1].mean():.4f} | ' \
              f'Mean GMean: {all_metrics[:n_bags, 2].mean():.4f}'
        print(log)
        tpr, tnr, gmean, log = log_metrics(test_labels, 
                    (all_preds[:n_bags, :].mean(axis=0) > 0.5).astype(int), 
                    run, feature_ranker, classifier_name, 
                    len(features), f'Ensemble of {n_bags} bags')
        print(log)

    print()       
    
    
def classify_feature_rank(DataCV_dir, Bags_dir, FeatRanking_dir, classifiers, params, 
                          dataset, feature_set, treatment, 
                          pval_pos_threshold, num_bags, num_runs,
                          num_top_features, test_all_features=False):
    
    train = load_mat(DataCV_dir/f'{feature_set}/{dataset}_{treatment}'
                                f'({pval_pos_threshold}).trn.mat')
    test = load_mat(DataCV_dir/f'{feature_set}/{dataset}_{treatment}'
                               f'({pval_pos_threshold}).tst.mat')
    
    feat_rank = pd.read_csv(FeatRanking_dir/f'{feature_set}/{dataset}_{treatment}'
                                            f'({pval_pos_threshold}).csv', sep=';')
    
    #if test_all_features:
        # add all features (6 metadata columns)
    #    num_top_features += [train.shape[1] - 6]
    
    trn_neg = train[train.Target == 0].reset_index(drop=True)
    trn_pos = train[train.Target == 1].reset_index(drop=True)
    
    for run in num_runs:
        # [1, 2, 3, 4, 5]
        print(f'Run: {run}') 
        bags = loadmat(Bags_dir/f'{dataset}({pval_pos_threshold})_Bags{run}.mat')
        # Sanity check
        n_samples = 6 if treatment == 'All' else 3
        assert bags['NegGenePairsBag'].max() == len(train[train.Target==0]) // n_samples
        assert bags['PosGenePairsBag'].max() == len(train[train.Target==1]) // n_samples
        bag_idx = lambda x: range(n_samples * (x-1), n_samples * x)
        
        feat_rank_rows = feat_rank[feat_rank.Run == run]
        
        for classifier, param in zip(classifiers, params):
                # [LGBMClassifier, LinearSVC, RandomForestClassifier]
                classifier_name = str(classifier).strip("<>'").split('.')[-1]
                print(' '*6, f'Classifier: {classifier_name}')
        
            for idx, feat_rank_row in feat_rank_rows.iterrows():
                # [Entropy, Ttest, Brattacharyya, Wilcoxon]
                print(' '*3, f'Feature ranking method: {feat_rank_row.Method}')             
                
                for num_top_feats in num_top_features:
                    # [2, 4, 6, 7, 10, 12] matlab index starts at 1
                    features = [col - 1 for col in feat_rank_row[2:2+num_top_feats]]
                    print(' '*6, f'Features: {features}')
                    
                    test_data = test.iloc[:,5:-1].iloc[:,features]
                    test_labels = test.Target
 
                    evaluate_bags(trn_neg, trn_pos, test_data, test_labels, 
                                  run, classifier, classifier_name, param, 
                                  feat_rank_row.Method, features, bags, bag_idx, num_bags)    
    
            if test_all_features:
                print(' '*3, f'All features')
                # use all features (6 metadata columns)
                num_top_feats = [train.shape[1] - 6]
                # [2, 4, 6, 7, 10, 12] matlab index starts at 1
                features = list(range(num_top_feats))
                print(' '*6, f'Features: {features}')

                test_data = test.iloc[:,5:-1].iloc[:,features]
                test_labels = test.Target

                evaluate_bags(trn_neg, trn_pos, test_data, test_labels, 
                              run, classifier, classifier_name, param, 
                              feat_rank_row.Method, features, bags, bag_idx, num_bags)
                    
                    
                          