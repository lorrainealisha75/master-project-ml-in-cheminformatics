import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from impyute.imputation.cs import em
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

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


def get_rf_classifier_model():
    return RandomForestClassifier(min_samples_leaf=15, n_estimators=250, max_depth=20, random_state=0)


def get_gb_classifier_model():
    return GradientBoostingClassifier(n_estimators=150, learning_rate=1, max_features=30, max_depth=50, random_state=0)


def get_svm_classifier_model():
    return SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,\
               decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',\
               max_iter=-1, probability=False, random_state=None, shrinking=True,\
               tol=0.001, verbose=False)


def plot_feature_importance(feature_importances):
    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()

    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(feature_importances.index[:15]))),
            feature_importances['normalized_importance'].head(15),
            align='center', edgecolor='k')

    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(feature_importances.index[:15]))))
    ax.set_yticklabels(feature_importances['feature'].head(15), size=12)

    # Plot labeling
    plt.xlabel('Normalized Importance', size=16);
    plt.title('Feature Importance', size=18)
    plt.show()


def plot_cumulative_importance(feature_importances, cumulative_importance):
    # Cumulative importance plot
    plt.figure(figsize=(6, 4))
    plt.plot(list(range(1, len(feature_importances) + 1)), feature_importances['cumulative_importance'], 'r-')
    plt.xlabel('Number of Features', size=14);
    plt.ylabel('Cumulative Importance', size=14);
    plt.title('Cumulative Feature Importance', size=16);

    if cumulative_importance:
        # Index of minimum number of features needed for cumulative importance threshold
        importance_index = np.min(np.where(feature_importances['cumulative_importance'] > cumulative_importance))
        print('\n%d features required for %0.2f of cumulative importance' % (importance_index + 1, cumulative_importance))
        plt.vlines(x=importance_index + 1, ymin=0, ymax=1, linestyles='--', colors='blue')
        plt.show()


def plot_collinearity_matrix(correlation_mat):
    sns.heatmap(correlation_mat)


def print_prediction_metrics(y_true, y_pred):
    print("\nPrecision is: " + str(precision_score(y_true, y_pred)))

    print("\nRecall is: " + str(recall_score(y_true, y_pred)))

    print("\nf1 score is: " + str(f1_score(y_true, y_pred)))


def plot_roc_curve(y_true, y_pred):
    print("\nROC-AUC score:")
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_pred)
        roc_auc[i] = auc(fpr[i], tpr[i])

    print(roc_auc_score(y_true, y_pred))
    plt.figure()
    plt.plot(fpr[1], tpr[1], color='darkorange',\
             label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    print("\nConfusion Matrix: ")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.ylabel("True Values")
    plt.xlabel("Predicted Values")
    plt.title("Confusion Matrix")
    plt.show()


# This is the output from the rdkit descriptor calculator tool on Galaxy
filename1 = "molecular-descriptors-mordred.tabular"

# This is the initial dataset that contains a list of compounds \
# (in SMILES format) and a label in the last column (0 or 1) which \
# indicates if the compound is active against the estrogen nuclear receptor
filename2 = "er-data.smi"

# Import data into a pandas dataframe
descriptors = pd.read_csv(filename1, sep='\t', dtype='O')
dataset = pd.read_csv(filename2, sep='\t', dtype='O')

# Keep only columns that represent features i.e the molecular descriptors
X = descriptors
y = dataset['label']

print("\nNumber of columns in the original dataset: " + str(len(X.columns)))

# Change the datatype of the features to float and the labels to int
X = X.astype("float")
y = y.astype("int")

# Replace inf to nan
X = X.replace([np.inf], np.nan)
y = y.replace([np.inf], np.nan)

col_stats = calculate_column_statistics(X)
columns_to_drop.update(col_stats)


# Impute nan values to the mean of the respective columns using SimpleImputer
imputer = SimpleImputer(add_indicator=False, copy=True, fill_value=None,\
                        missing_values=np.nan, strategy='mean', verbose=0)

'''
# Impute nan values to the mean of the respective columns using IterativeImputer
imputer = IterativeImputer(add_indicator=False, estimator=None,\
                           imputation_order='ascending', initial_strategy='mean',\
                           max_iter=10, max_value=None, min_value=None,\
                           missing_values=np.nan, n_nearest_features=None,\
                           random_state=0, sample_posterior=False, tol=0.001,\
                           verbose=0)
'''

X = imputer.fit_transform(X)

'''
# Imputes given data using expectation maximization
X = em(X, loops=50)
'''

# Create a pandas df again from the output of the imputer
X = pd.DataFrame(data=X[0:, 0:])

# Extract feature names
feature_names = list(X.columns)

# Convert to np array
features = np.array(X)
labels = np.array(y).reshape((-1,))

print('\nTraining Gradient Boosting Model')

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

# Plotting statistics
#plot_collinearity_matrix(corr_mat)

# Plot feature importance
#plot_feature_importance(feature_importances)

# Plot cumulative importance
#plot_cumulative_importance(feature_importances, cumulative_importance)

# Separate data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

# Classifier algorithm
# 0 - Random Forest
# 1 - Gradient Boosting
# 3 - SVM classifier
options = {0: get_rf_classifier_model,\
           1: get_gb_classifier_model,\
           2: get_svm_classifier_model
           }

classifier_type = 1

# Fit the training data on a Gradient Boost classifier
classifier_model = options[classifier_type]()

classifier_model.fit(X_train, y_train)

y_pred = classifier_model.predict(X_test)

print("\nAccuracy: " + str(classifier_model.score(X_test, y_test)))

# Get Precision, Recall, f1 score and confusion matrix
print_prediction_metrics(y_test, y_pred)

# Plot ROC curve and confusion matrix
plot_confusion_matrix(y_test, y_pred)
plot_roc_curve(y_test, y_pred)
