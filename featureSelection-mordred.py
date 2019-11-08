# Feature selection (mordred)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier
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


# Apply SelectKBest class to extract top n best features
bestfeatures = SelectKBest(score_func=f_classif, k=20)

# Apply ExtraTreesClassifier class to get ranking of the features
#bestfeatures = ExtraTreesClassifier()

fit = bestfeatures.fit(X, y)

scores = pd.DataFrame(fit.scores_)
#scores = pd.DataFrame(fit.feature_importances_)
columns = pd.DataFrame(X.columns)

# Concat two dataframes
feature_scores = pd.concat([columns, scores], axis=1)
feature_scores.columns = ['Feature', 'Score']
n_largest = feature_scores.nlargest(20, 'Score')
print(n_largest)
features = n_largest['Feature'].values.tolist()

# Extract selected features out of the dataset.
X = X[features]

# Separate data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

file_names = ['X_train', 'X_test', 'y_train', 'y_test']
datasets = [X_train, X_test, y_train, y_test]

# Convert the train and test sets to csv files
for dataset, file_name in zip(datasets, file_names):
    dataset.to_csv(file_name, sep='\t', header=None, index=False)


# Fit the training data on a random forest classifier
rForest = RandomForestClassifier(min_samples_leaf=15, n_estimators=250, max_depth=20, random_state=0)
rForest.fit(X_train, y_train)

score = rForest.score(X_test, y_test)
print("Quality of the model: " + str(score))
