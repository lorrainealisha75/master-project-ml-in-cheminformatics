# Task 2: Using rdkit

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# This is the output from the rdkit descriptor calculator tool on Galaxy
filename1 = "molecular-descriptors-rdkit.tabular"

# This is the initial dataset that contains a list of compounds \
# (in SMILES format) and a label in the last column (0 or 1) which \
# indicates if the compound is active against the estrogen nuclear receptor
filename2 = "er-data.smi"

# Import data into a pandas dataframe
p_df1 = pd.read_csv(filename1, sep='\t', dtype='O')
p_df2 = pd.read_csv(filename2, sep='\t', dtype='O')

# Join the datasets using full outer join on the first column in both datasets
df_merged = p_df1.merge(p_df2, on='MoleculeID', how='outer')

# Keep only columns that represent features i.e the molecular descriptors
X = df_merged[df_merged.columns[~df_merged.columns.isin(['MoleculeID', 'Receptor', 'label'])]]
y = df_merged[df_merged.columns[df_merged.columns.isin(['label'])]]

# Change the datatype of the features to float and the labels to int
X = X.astype("float32")
y = y.astype("int32")

# Replace inf to nan
X = X.replace([np.inf], np.nan)
y = y.replace([np.inf], np.nan)

# Convert nan values to 0
X.fillna(value=0, inplace=True)
y.fillna(value=0, inplace=True)

# Separate data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

file_names = ['X_train', 'X_test', 'y_train', 'y_test']
datasets = [X_train, X_test, y_train, y_test]

# Convert the train and test sets to csv files
for dataset, file_name in zip(datasets, file_names):
    dataset.to_csv(file_name, sep='\t', header=None, index=False)


# Fit the training data on a random forest classifier
classifier_model = RandomForestClassifier(min_samples_leaf=15, n_estimators=250, max_depth=20, random_state=0)

# Fit the training data on a Gradient Boost classifier
#classifier_model = GradientBoostingClassifier(n_estimators=150, learning_rate=1, max_features=30, max_depth=50, random_state=0)

classifier_model.fit(X_train, y_train)

print("Quality of the model: " + str(classifier_model.score(X_test, y_test)))
