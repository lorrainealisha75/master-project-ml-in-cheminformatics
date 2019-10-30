# Task 2

import pandas as pd
from sklearn.model_selection import train_test_split

# This is the output from the rdkit descriptor calculator tool on Galaxy
filename1 = "molecular-descriptors.tabular"

# This is the initial dataset that contains a list of compounds \
# (in SMILES format) and a label in the last column (0 or 1) which \
# indicates if the compound is active against the estrogen nuclear receptor
filename2 = "er-data.smi"

# Import data into a pandas dataframe
p_df1 = pd.read_csv(filename1, sep='\t', header=None, dtype='O')
p_df2 = pd.read_csv(filename2, sep='\t', header=None, dtype='O')

# Join the datasets using full outer join on the first column in both datasets
df_merged = p_df1.merge(p_df2, on=0, how='outer')

# Keep only columns that represent features i.e the molecular descriptors
X = df_merged[df_merged.columns[~df_merged.columns.isin([0, '1_y', '2_y'])]]
y = df_merged[df_merged.columns[df_merged.columns.isin(['2_y'])]]

# Separate data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

file_names = ['X_train', 'X_test', 'y_train', 'y_test']
datasets = [X_train, X_test, y_train, y_test]

# Convert the train and test sets to csv files
for dataset, file_name in zip(datasets, file_names):
    dataset.to_csv(file_name, sep='\t', header=None, index=False)




