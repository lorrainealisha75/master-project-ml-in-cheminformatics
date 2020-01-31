import pandas as pd
import numpy as np
import lightgbm as lgbm
from skopt import BayesSearchCV
from sklearn.model_selection import KFold


def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""

    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

    # Get current parameters and the best parameters
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest MSE: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))

    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name + "_cv_results.csv")


# This is the output from the rdkit descriptor calculator tool on Galaxy
filename1 = "data/mordred-nr-ppar-gamma.tabular"

# This is the initial dataset that contains a list of compounds \
# (in SMILES format) and a label in the last column (0 or 1) which \
# indicates if the compound is active against the estrogen nuclear receptor
filename2 = "data/nr-ppar-gamma.smiles"

# Import data into a pandas dataframe
descriptors = pd.read_csv(filename1, sep='\t', header=None, dtype='O')
dataset = pd.read_csv(filename2, sep='\t', dtype='O')

# Keep only columns that represent features i.e the molecular descriptors
X = descriptors
y = dataset['label']

# Change the datatype of the features to float and the labels to int
X = X.astype("float32")
y = y.astype("int64")

# Replace inf to nan
X = X.replace([np.inf], np.nan)
y = y.replace([np.inf], np.nan)


# Convert nan values to 0
X.fillna(value=0, inplace=True)
y.fillna(value=0, inplace=True)

bayes_cv_tuner = BayesSearchCV(
    estimator=lgbm.LGBMClassifier(objective=None, boosting_type='gbdt', subsample=0.6143),
    search_spaces={
        'learning_rate': (0.001, 1.0, 'log-uniform'),
        'num_leaves': (10, 100),
        'max_depth': (0, 50),
        'min_child_samples': (0, 50),
        'subsample_freq': (0, 10),
        'min_child_weight': (0, 10),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1, 'log-uniform'),
        'scale_pos_weight': (1e-9, 500, 'log-uniform'),
        'n_estimators': (50, 500),
        'max_bin': (100, 1000),
        'num_iterations': (50, 500),
        'nthread': (1, 10)
    },
    scoring='neg_mean_squared_log_error',
    cv=KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    ),
    n_jobs=1,
    n_iter=25,
    verbose=0,
    refit=True,
    random_state=42
)

print("\nFit the model\n")
# Fit the model
result = bayes_cv_tuner.fit(X, y, callback=status_print)