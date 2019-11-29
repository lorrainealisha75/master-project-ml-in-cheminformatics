import pandas as pd
import numpy as np

import seaborn as sns
from math import factorial

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.impute import SimpleImputer

from sklearn.ensemble import GradientBoostingClassifier


# List of all columns to be removed from the dataset
columns_to_drop = set()

# Number of epochs on the training set
n_iterations = 1

# Total cumulative importance of the features
cumulative_importance = 0.95


def calculate_column_statistics(X):

    cols_to_remove = set()

    # Count the number of nan values in every column
    nan_columns = np.where((X.isna().sum() / X.shape[0]) > 0.6)[0].tolist()

    cols_to_remove.update(nan_columns)

    # Get a list of all columns with a single unique value and remove them
    single_unique = np.where(X.nunique() == 1)[0].tolist()

    cols_to_remove.update(single_unique)

    # Get a list of all columns with 0 stddev
    constant_columns = np.where(np.std(X, axis=0) == 0)[0].tolist()

    cols_to_remove.update(constant_columns)

    print("\nNumber of columns after performing column statistics: " + str(len(X.columns) - len(cols_to_remove)))

    return cols_to_remove


def remove_collinear_features(X):
    correlation_matrix = X.corr()

    correlation_matrix_with_avg = correlation_matrix.copy()

    # Add a new column with the row-wise average
    correlation_matrix_with_avg['average'] = correlation_matrix_with_avg.mean(numeric_only=True, axis=1)

    sns.heatmap(correlation_matrix)

    columns = np.full((correlation_matrix.shape[0],), True, dtype=bool)

    for i in range(correlation_matrix.shape[0]):
        for j in range(i + 1, correlation_matrix.shape[0]):
            # If the collinearity coefficient is too high, remove one of the features
            if correlation_matrix.iloc[i, j] >= 0.9 and columns[j]:
                # Remove the feature that has a higher collinearity coefficient with all other features
                if correlation_matrix_with_avg.iloc[j]['average'] > correlation_matrix_with_avg.iloc[i]['average']:
                    columns[j] = False
                else:
                    columns[i] = False

    selected_columns = X.columns[columns]

    print("\nNumber of columns after removing columns with high correlation: " + str(len(X.columns)))
    return X[selected_columns], correlation_matrix


def get_feature_importance(iterations, X, y):
    # Empty array for feature importances
    feature_importance = np.zeros(len(feature_names))

    # Iterate through each fold
    for _ in range(iterations):
        # model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, verbose=-1)
        model = GradientBoostingClassifier(n_estimators=150, learning_rate=1, max_features=30, max_depth=50,
                                           random_state=0)
        model.fit(X, y)

        # Record the feature importances
        feature_importance += model.feature_importances_ / iterations

    return feature_importance


def process_feature_importance(feature_import):
    # Sort features according to importance
    feature_import = feature_import.sort_values('importance', ascending=False).reset_index(drop=True)

    # Normalize the feature importances to add up to one
    feature_import['normalized_importance'] = feature_import['importance'] / feature_import[
        'importance'].sum()
    return feature_import


def zero_importance_features(feature_importance):
    zero_imp_features = feature_importance[feature_importance['importance'] == 0.0]
    return set(zero_imp_features['feature'])


def low_importance_feature(cumulative_imp, feature_importance):

    # Make sure most important features are on top
    feature_importance = feature_importance.sort_values('cumulative_importance')

    # Identify the features not needed to reach the cumulative_importance
    low_importance_features = feature_importance[feature_importance['cumulative_importance'] > cumulative_imp]

    return set(low_importance_features['feature'])


def remove_columns(X, columns):
    # Get the indices for the given column names
    to_drop_index = [X.columns.get_loc(col) for col in columns]

    # Drop relevant columns
    X.drop(X.columns[to_drop_index], axis=1, inplace=True)
    return X


# This is the output from the rdkit descriptor calculator tool on Galaxy
filename1 = "molecular-descriptors-mordred.tabular"

# This is the initial dataset that contains a list of compounds \
# (in SMILES format) and a label in the last column (0 or 1) which \
# indicates if the compound is active against the estrogen nuclear receptor
filename2 = "er-data.smi"

# Import data into a pandas dataframe
descriptors = pd.read_csv(filename1, sep='\t', dtype='O')
dataset = pd.read_csv(filename2, sep='\t', dtype='O')


X_orig = descriptors
y_orig = dataset['label']


# Change the datatype of the features to float and the labels to int
X_orig = X_orig.astype("float")
y_orig = y_orig.astype("int")

# Replace inf to nan
X_orig = X_orig.replace([np.inf], np.nan)
y_orig = y_orig.replace([np.inf], np.nan)

print("\nNumber of features in the original dataset: " + str(len(X_orig.columns)))

headers = X_orig.columns.to_numpy()

X_orig.columns = range(X_orig.shape[1])
X_orig = X_orig.to_numpy()
y_orig = y_orig.to_numpy()

feature_sets = []
num_of_splits = 5
i = 1

skf = StratifiedKFold(n_splits=num_of_splits, random_state=None, shuffle=False)
for train_index, test_index in skf.split(X_orig, y_orig):
    X_train, X_test = X_orig[train_index], X_orig[test_index]
    y_train, y_test = y_orig[train_index], y_orig[test_index]

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    X = pd.DataFrame(data=X[0:, 0:])
    y = pd.DataFrame(data=y[0:])

    col_stats = calculate_column_statistics(X)
    columns_to_drop.update(col_stats)

    # Impute nan values to the mean of the respective columns using SimpleImputer
    imputer = SimpleImputer(add_indicator=False, copy=True, fill_value=None, \
                            missing_values=np.nan, strategy='mean', verbose=0)

    X = imputer.fit_transform(X)

    # Create a pandas df again from the output of the imputer
    X = pd.DataFrame(data=X[0:, 0:])

    # Extract feature names
    feature_names = list(X.columns)

    # Convert to np array
    features = np.array(X)
    labels = np.array(y).reshape((-1,))

    feature_importance_values = get_feature_importance(n_iterations, features, labels)

    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    feature_importances = process_feature_importance(feature_importances)
    feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])

    # Extract the features with zero importance
    record_zero_importance = zero_importance_features(feature_importances)

    # Add zero importance features to the list of columns to be removed
    columns_to_drop.update(record_zero_importance)

    # Extract the features with low importance
    record_low_importance = low_importance_feature(cumulative_importance, feature_importances)

    # Add low importance features to the list of columns to be removed
    columns_to_drop.update(record_low_importance)

    X = remove_columns(X, columns_to_drop)

    # Remove features that display high collinear coefficient
    X, corr_mat = remove_collinear_features(X)

    print("\nNumber of columns after feature selection: " + str(len(X.columns)))

    features = headers[X.columns]
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
