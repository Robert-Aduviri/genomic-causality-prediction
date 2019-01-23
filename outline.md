### Dimensions

- [inhibitor, feature set, treatment, run, classifier, feature ranking method, num top features, bags/ensemble]

- Visualization: [inhibitor, feature set, treatment, classifier, feature ranking method], best cv score of [run, num top features] combinations, ensemble and bags average

### Experimentation outline

Preprocessing
- Rankings
- Bags

1. Load train and test set corresponding to a [inhibitor, feature set, treatment] combination
2. For each classifier:
    2.1. Do cross validation loop (k=5) (train => trn, val):
        2.1.1. For each feature ranking method:
            2.1.1.1. Calculate feature ranking for trn dataset
                2.1.1.1.1. Get 100 random bags for feature ranking
            2.1.1.2. For each num top features:
                2.1.1.2.1. Select top features from trn and test
                2.1.1.2.2. Get 100 random bags from trn
                2.1.1.2.3. For each bag:
                    2.1.1.2.3.1. Fit model with bag
                    2.1.1.2.3.2. Predict data from val and test
                    2.1.1.2.3.3. Report bag, val and test metrics
                    2.1.1.2.3.4. Save bag, val and test predictions
                2.1.1.2.4. For every [100] bags:
                    2.1.1.2.4. Report mean tpr, tnr and gmean 
        2.1.2. Using all features, perform the same steps from (2.1.2.2)
    2.2. Average all scores across all validation folds
    2.3. Select best result among all num_features / ensemble sizes ([inhibitor, feature set, treatment, classifier, feature ranker + all features])
    2.4. For each best combination, retrain on all train set and report metrics on test set
        2.4.1. For every [25, 50, 75, 100] bags:
            2.4.1.1. Perform voting ensembling with predictions
            2.4.1.2. Report ensemble tpr, tnr and gmean