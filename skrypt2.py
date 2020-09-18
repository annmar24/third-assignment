import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score


with open('survey_results_public.csv', 'rb') as f:
    # Read and prepare data
    df = pd.read_csv(f, usecols=['Respondent', 'Hobbyist', 'Age', 'Student', 'YearsCode', 'YearsCodePro'], index_col='Respondent')

    df.dropna(inplace=True)
    df.replace(to_replace={
            "Yes": "1",
            "No": "0",
            "Yes, part-time": "1",
            "Yes, full-time": "1",
            "Less than 1 year": "0",
            "More than 50 years": "51"
        },
        inplace=True)
    df = df.astype('int64')

    # Logistic reggression
    clf = LogisticRegression()
    x_train = df[['Age', 'Student', 'YearsCode', 'YearsCodePro']]
    y_train = df['Hobbyist']
    clf.fit(x_train, y_train)
    y_train_pred = clf.predict(x_train)


    # Accuracy
    clf_accuracy = accuracy_score(y_train, y_train_pred)

    # Sensitivity
    clf_sensitivity = recall_score(y_train, y_train_pred)

    # Confusion matrix and specificity
    conf_matrix = confusion_matrix(y_train, y_train_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    clf_specificity = tn / (tn + fp)

    # zad. 2
    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(df[['Age', 'Student', 'YearsCode', 'YearsCodePro']], df['Hobbyist'], test_size=0.25, random_state=5)

    # Train data on train set
    clf_split = LogisticRegression()
    clf_split.fit(x_train, y_train)

    # Predict on test set
    y_test_pred = clf_split.predict(x_test)

     # Accuracy
    clf_test_accuracy = accuracy_score(y_test, y_test_pred)

    # Sensitivity
    clf_test_sensitivity = recall_score(y_test, y_test_pred)

    # Confusion matrix and specificity
    conf_test_matrix = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    clf_test_specificity = tn / (tn + fp)


