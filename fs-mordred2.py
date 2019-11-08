# Using Backward Elimination

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# This is the output from the rdkit descriptor calculator tool on Galaxy
filename1 = "molecular-descriptors-mordred.tabular"

# This is the initial dataset that contains a list of compounds \
# (in SMILES format) and a label in the last column (0 or 1) which \
# indicates if the compound is active against the estrogen nuclear receptor
filename2 = "er-data.smi"

# Import data into a pandas dataframe
p_df1 = pd.read_csv(filename1, sep='\t', dtype='O')
p_df2 = pd.read_csv(filename2, sep='\t', dtype='O')

# Keep only columns that represent features i.e the molecular descriptors
X = p_df1
y = p_df2['label']


# np.reshape(y, (X.shape[0], 1))
# Change the datatype of the features to float and the labels to int
X = X.astype("float32")
y = y.astype("int32")

# Replace inf to nan
X = X.replace([np.inf], np.nan)
y = y.replace([np.inf], np.nan)

# Convert nan values to 0
X.fillna(value=0, inplace=True)
y.fillna(value=0, inplace=True)


#Backward Elimination
cols = list(X.columns)
pmax = 1

# Select 30 features
while len(cols) > 30:
    p = []
    X_1 = X[cols]
    # Add column of 1's
    X_1 = sm.add_constant(X_1)

    # Use ordinary least squares
    model = sm.OLS(y, X_1).fit()
    p = pd.Series(model.pvalues.values[1:], index=cols)
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if pmax > 0.05:
        cols.remove(feature_with_p_max)
    else:
        break
features = cols

# print(features)
X = X[features]


# Separate data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

# y_train = np.ravel(y_train)  # .reshape(X_train.length, 1)
# y_test = np.ravel(y_test)  # .reshape(X_test.length, 1)

file_names = ['X_train', 'X_test', 'y_train', 'y_test']
datasets = [X_train, X_test, y_train, y_test]

# Convert the train and test sets to csv files
for dataset, file_name in zip(datasets, file_names):
    dataset.to_csv(file_name, sep='\t', header=None, index=False)


# Fit the training data on a random forest classifier
rForest = RandomForestClassifier(min_samples_leaf=5, n_estimators=100, max_depth=10, random_state=0)
rForest.fit(X_train, y_train)

score = rForest.score(X_test, y_test)
print("Quality of the model: " + str(score))