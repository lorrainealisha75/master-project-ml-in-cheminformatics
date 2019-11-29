import pandas as pd
import numpy as np

from math import factorial
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFdr, SelectPercentile

from sklearn.model_selection import StratifiedKFold

num_features = 30
num_of_splits = 5
perc = 10


def select_k_best():
    return SelectKBest(mutual_info_classif, k=num_features)


def select_percentile():
    return SelectPercentile(mutual_info_classif, percentile=perc)


# This is the output from the rdkit descriptor calculator tool on Galaxy
filename1 = "molecular-descriptors-mordred.tabular"

# This is the initial dataset that contains a list of compounds \
# (in SMILES format) and a label in the last column (0 or 1) which \
# indicates if the compound is active against the estrogen nuclear receptor
filename2 = "er-data.smi"

# Import data into a pandas dataframe
descriptors = pd.read_csv(filename1, sep='\t', dtype='O')
dataset = pd.read_csv(filename2, sep='\t', dtype='O')


X = descriptors
y = dataset['label']


# Change the datatype of the features to float and the labels to int
X = X.astype("float")
y = y.astype("int")

# Replace inf to nan
X = X.replace([np.inf], np.nan)
y = y.replace([np.inf], np.nan)

# Convert nan values to 0
X.fillna(value=0, inplace=True)
y.fillna(value=0, inplace=True)

headers = X.columns.to_numpy()

X.columns = range(X.shape[1])
X = X.to_numpy()
y = y.to_numpy()

feature_sets = []
i = 1

options = {0: select_k_best,\
           1: select_percentile
           }

fs_method = 1

skf = StratifiedKFold(n_splits=num_of_splits, random_state=None, shuffle=False)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    fs_model = options[fs_method]()
    fs_model.fit_transform(X, y)
    support = fs_model.get_support()
    features = headers[support]
    feature_sets.append(features)
    print("\nSet " + str(i) + ":")
    print("\nLength: " + str(len(features)))
    i += 1
    print(features)

sum = 0
for i in range(num_of_splits):
    for j in range(i + 1, num_of_splits):
        fs_len = max(abs(len(set(feature_sets[i]) - set(feature_sets[j]))), abs(len((set(feature_sets[j]) - set(feature_sets[i])))))
        print("\nLength difference between set " + str(i + 1) + " and set " + str(j + 1) + " is "\
              + str(fs_len))
        sum += (fs_len / (max(len(feature_sets[i]), len(feature_sets[j]))))*100

ncr = factorial(i + 1) / factorial(2) / factorial((i + 1) - 2)
print("\nOn an average, " + str(100 - sum/ncr) + "% of the features are same between folds")


