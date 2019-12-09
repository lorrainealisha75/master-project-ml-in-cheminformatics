import pandas as pd
import numpy as np
import gc

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgbm
from sklearn.svm import SVC
from boruta import BorutaPy

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

early_stopping = True
eval_metric = "auc"

def get_rf_classifier_model():
    return RandomForestClassifier(min_samples_leaf=15, n_estimators=250, max_depth=20, random_state=0)


def get_gb_classifier_model():
    return GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_features=30, max_depth=50, random_state=0)


def get_lightgbm_classifier_model():
    return lgbm.LGBMClassifier(n_estimators=1000, learning_rate=0.2, verbose=-1, is_unbalance=True)

def get_svm_classifier_model():
    return SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,\
               decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',\
               max_iter=-1, probability=False, random_state=None, shrinking=True,\
               tol=0.001, verbose=False)


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
filename1 = "molecular-descriptors-1-mordred.tabular"

# This is the initial dataset that contains a list of compounds \
# (in SMILES format) and a label in the last column (0 or 1) which \
# indicates if the compound is active against the estrogen nuclear receptor
filename2 = "er-data.smi"

# Import data into a pandas dataframe
descriptors = pd.read_csv(filename1, sep='\t', header=None, dtype='O')
dataset = pd.read_csv(filename2, sep='\t', dtype='O')

# Keep only columns that represent features i.e the molecular descriptors
X = descriptors
y = dataset['label']

print("\nNumber of features before selection: ", len(X.columns))

# np.reshape(y, (X.shape[0], 1))
# Change the datatype of the features to float and the labels to int
X = X.astype("float32")
y = y.astype("int64")

# Replace inf to nan
X = X.replace([np.inf], np.nan)
y = y.replace([np.inf], np.nan)


# Convert nan values to 0
X.fillna(value=0, inplace=True)
y.fillna(value=0, inplace=True)


# Classifier algorithm
# 0 - Random Forest
# 1 - Gradient Boosting
# 3 - SVM classifier
options = {0: get_rf_classifier_model,\
           1: get_gb_classifier_model,\
           2: get_lightgbm_classifier_model,\
           3: get_svm_classifier_model
           }

classifier_type = 0

# Fit the training data on a Gradient Boost classifier
classifier_model = options[classifier_type]()

boruta_feature_selector = BorutaPy(classifier_model, n_estimators='auto', verbose=2, random_state=4242, max_iter=25,\
                                   perc=90)
boruta_feature_selector.fit(X.to_numpy(), y.to_numpy())

X_filtered = boruta_feature_selector.transform(X.to_numpy())

print("\nNumber of features after selection: ", X_filtered.shape[1])

classifier_type = 2

# Fit the training data on a Gradient Boost classifier
classifier_model = options[classifier_type]()

print("\nTrain the model")
# Separate data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, train_size=0.8, shuffle=True)

if early_stopping:

    # Separate data into train and validation
    train_features, valid_features, train_labels, valid_labels = train_test_split(X_train, y_train, train_size=0.8)

    # Train the model with early stopping
    classifier_model.fit(train_features, train_labels, eval_metric=eval_metric,
              eval_set=[(valid_features, valid_labels)],
              early_stopping_rounds=100, verbose=-1)

    # Clean up memory
    gc.enable()
    del train_features, train_labels, valid_features, valid_labels
    gc.collect()

else:
    classifier_model.fit(X_train, y_train)

y_pred = classifier_model.predict(X_test)

print("\nAccuracy: " + str(classifier_model.score(X_test, y_test)))

# Get Precision, Recall, f1 score and confusion matrix
print_prediction_metrics(y_test, y_pred)

# Plot ROC curve and confusion matrix
plot_confusion_matrix(y_test, y_pred)
plot_roc_curve(y_test, y_pred)

