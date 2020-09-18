import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import plot_confusion_matrix

data_file = os.path.join('data', 'train.tsv')
test_file = os.path.join('data', 'test.tsv')
results_file = os.path.join('data', 'results.tsv')

# Read training set data file as dataframe
df_names = ['Occupancy', 'Date', 'Temperature', 'Humidity',
            'Light', 'CO2', 'HumidityRatio']

df = pd.read_csv(data_file, sep='\t', names=df_names)
df = df.dropna()

## Calculations for training data set

# Logistic regression classifier on one independent variable - Temperature
clf = LogisticRegression()
x_train = df[['Temperature']]
y_train = df.Occupancy
clf.fit(x_train, y_train)
y_train_pred = clf.predict(x_train)

# Calculating training set accuracy for logistic regression model on Temperature
clf_accuracy = accuracy_score(y_train, y_train_pred)

# Calculating training set sensitivity for logistic regression model on Temperature
clf_sensitivity = recall_score(y_train, y_train_pred)

# Calculating training set specificity for logistic regression model on Temperature
# Confusion matrix
conf_matrix = confusion_matrix(y_train, y_train_pred)
tn, fp, fn, tp = conf_matrix.ravel()
clf_specificity = tn / (tn + fp)

# Logistic regression classifier on all variables (date independent)
clf_all = LogisticRegression()
x_train_all = df[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]
clf_all.fit(x_train_all, y_train);
y_train_pred_all = clf_all.predict(x_train_all)

# Calculating training set accuracy for logistic regression model on all variables
clf_all_accuracy = accuracy_score(y_train, y_train_pred_all)

# Calculating training set sensitivity for logistic regression model on all variables
clf_all_sensitivity = recall_score(y_train, y_train_pred_all)

# Calculating training set specifity for logistic regression model on all variables
# Confusion matrix
conf_matrix = confusion_matrix(y_train, y_train_pred_all)
tn, fp, fn, tp = conf_matrix.ravel()
clf_all_specifity = tn / (tn + fp)

## Testing data set

# Read testing set data file as dataframe
df_test_names = ['Date', 'Temperature', 'Humidity',
                 'Light', 'CO2', 'HumidityRatio']
x_column_names = ['Temperature', 'Humidity',
                 'Light', 'CO2', 'HumidityRatio']
df_test = pd.read_csv(test_file, sep='\t', names=df_test_names, usecols=x_column_names)
df_test = df_test.dropna()

# Read results
df_results = pd.read_csv(results_file, sep='\t', names=['y'])
df_results['y'] = df_results['y'].astype('category')

y_true = df_results['y']

# Logistic regression classifier on Temperature variable - testing data
x_test = df_test[['Temperature']]
y_test_pred = clf.predict(x_test)

# Calculating testing set accuracy for logistic regression model on Temperature
clf_test_accuracy = accuracy_score(y_true, y_test_pred)

# Calculating testing set sensitivity for logistic regression model on Temperature
clf_test_sensitivity = recall_score(y_true, y_test_pred)

# Calculating testing set specificity for logistic regression model on Temperature
# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_test_pred)
tn, fp, fn, tp = conf_matrix.ravel()
clf_test_specificity = tn / (tn + fp)

# Logistic regression classifier on all variables - testing data
y_test_pred_all = clf_all.predict(df_test)

# Calculating accuracy for testing data set for all variables
clf_test_all_accuracy = accuracy_score(y_true, y_test_pred_all)

# Calculating testing set sensitivity for all variables
clf_test_all_sensitivity = recall_score(y_true, y_test_pred_all)

# Calculating testing set specificity for all variables
# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_test_pred_all)
tn, fp, fn, tp = conf_matrix.ravel()
clf_test_all_specifity = tn / (tn + fp)

# Save results to output file
output_file = os.path.join('data', 'out.tsv')
df = pd.DataFrame(y_test_pred, y_test_pred_all)
df.to_csv(output_file, index=False, header=False)


# zad.2
# F measure (f1 score)
# For one variable
f_one_train = f1_score(y_train, y_train_pred)
f_one_test = f1_score(y_true, y_test_pred)
# Someona should be in the room
f_beta_one_train_a = fbeta_score(y_train, y_train_pred, beta=0.5)
f_beta_one_test_a = fbeta_score(y_true, y_test_pred, beta=0.5)
# There shouldn't be anyone in the room
f_beta_one_train_b = fbeta_score(y_train, y_train_pred, beta=2.0)
f_beta_one_test_b = fbeta_score(y_true, y_test_pred, beta=2.0)

# F measure
# For all variables (except data)
f_all_train = f1_score(y_train, y_train_pred_all)
f_all_test = f1_score(y_true, y_test_pred_all)
# Someona should be in the room
f_beta_all_train_a = fbeta_score(y_train, y_train_pred_all, beta=0.5)
f_beta_all_test_a = fbeta_score(y_true, y_test_pred_all, beta=0.5)
# There shouldn't be anyone in the room
f_beta_all_train_b = fbeta_score(y_train, y_train_pred_all, beta=2.0)
f_beta_all_test_b = fbeta_score(y_true, y_test_pred_all, beta=2.0)
