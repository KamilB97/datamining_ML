import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

import pandas as pd
import numpy as np
import os
import pyreadstat
import seaborn as sns
import matplotlib.pyplot as plt
from sas7bdat import SAS7BDAT


def run_classifier():
    # Load data
    copper_wire_df, meta = pyreadstat.read_sas7bdat('copper_wire_bin.sas7bdat')
    copper_wire_df.drop('quality', inplace=True, axis=1)

    # Remove Nan and Infinite values (Choose one option)
    # RemoveNanAndInfWithDeletimgRecords(copper_wire_df)
    # RemoveNanAndInfWithMedian(copper_wire_df)
    RemoveNanAndInfWithMode(
        copper_wire_df)  # fill NaN and infinite values with the most frequently value occured //best

    # remove rows with negative values
    copper_wire_df = copper_wire_df[copper_wire_df.select_dtypes(include=[np.number]).ge(0).all(1)]

    ## Separating predictors and response
    y = copper_wire_df.iloc[:, 14]
    X = copper_wire_df.iloc[:, :14]

    ## K best features
    # X = SelectKBest(score_func=chi2, k=12).fit_transform(X, y)
    #PCA
    # X = perform_pca(X, copper_wire_df)

    ## Split for training and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    ##undersample // nice results
    # X_train, y_train = undersample(X_train, y_train)

    ##oversample // słabiutko
    # X_train, y_train = oversample(X_train, y_train)

    ## Feature selection
    ## RFE
    # X_test, X_train, y_test, y_train = recursiveFeatureElimination(X, y)

    ## model forward feature selection
    # https://pythonhealthcare.org/2020/01/04/feature-selection-2-model-forward-selection/

## some forward feature selection https://pythonhealthcare.org/2020/01/04/feature-selection-2-model-forward-selection/

    # Classifier training using Logistic regression
    logReg_classifier = LogisticRegression(solver='lbfgs', max_iter=1150)
    logReg_classifier.fit(X_train, y_train)
    y_predict_logReg = logReg_classifier.predict(X_test)
    # Check Classifier accuracy on the test data and see results
    print("Logistic Regression:")
    print("Accuracy:", accuracy_score(y_test, y_predict_logReg))
    print(classification_report(y_test, y_predict_logReg))
    print("confusion matrix:")
    print(confusion_matrix(y_test, y_predict_logReg))


def perform_pca(X, copper_wire_df):
    scaler = StandardScaler()
    scaler.fit(copper_wire_df)
    scaled_data = scaler.transform(copper_wire_df)
    pca = PCA(n_components=8)
    pca.fit(scaled_data)
    X = pca.transform(scaled_data)
    return X


def oversample(X_train, y_train):
    print("X_train before over: " + str(X_train.shape))
    ros = RandomOverSampler(random_state=8)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    print("X_train after oversample: " + str(X_train.shape))
    return X_train, y_train


def undersample(X_train, y_train):
    print("X_train before undersample: " + str(X_train.shape))
    rus = RandomUnderSampler(random_state=8, replacement=True)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    print("X_train after undersample: " + str(X_train.shape))
    return X_train, y_train

def recursiveFeatureElimination(X, y):
    # no of features
    nof_list = np.arange(1, 2) #set to 5 also gives 100%
    high_score = 0
    # Variable to store the optimum features
    nof = 0
    score_list = []
    for n in range(len(nof_list)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        model = SVC(kernel='linear', gamma='auto')
        rfe = RFE(model, nof_list[n])
        X_train_rfe = rfe.fit_transform(X_train, y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe, y_train)
        score = model.score(X_test_rfe, y_test)
        score_list.append(score)
        if (score > high_score):
            high_score = score
            nof = nof_list[n]
    print("Optimum number of features: %d" % nof)
    print("Score with %d features: %f" % (nof, high_score))
    return X_test, X_train, y_test, y_train


def RemoveNanAndInfWithDeletimgRecords(copper_wire_df):
    # Replace infinite with Nan
    print(copper_wire_df.shape)
    copper_wire_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Dropping all the rows with nan values
    copper_wire_df.dropna(inplace=True)
    print(copper_wire_df.shape)


def RemoveNanAndInfWithMedian(copper_wire_df):
    # fill NaN and infinite values with median
    copper_wire_df['sum_eddy_f1'] = copper_wire_df['sum_eddy_f1'].fillna(copper_wire_df['sum_eddy_f1'].median())
    copper_wire_df['sum_eddy_f2'] = copper_wire_df['sum_eddy_f2'].fillna(copper_wire_df['sum_eddy_f2'].median())
    copper_wire_df['sum_eddy_f3'] = copper_wire_df['sum_eddy_f3'].fillna(copper_wire_df['sum_eddy_f3'].median())
    copper_wire_df['sum_ferro_f1'] = copper_wire_df['sum_ferro_f1'].fillna(copper_wire_df['sum_ferro_f1'].median())
    copper_wire_df['sum_ferro_f2'] = copper_wire_df['sum_ferro_f2'].fillna(copper_wire_df['sum_ferro_f2'].median())
    copper_wire_df['sum_ferro_f3'] = copper_wire_df['sum_ferro_f3'].fillna(copper_wire_df['sum_ferro_f3'].median())
    copper_wire_df['size_min'] = copper_wire_df['size_min'].fillna(copper_wire_df['size_min'].median())
    copper_wire_df['size_max'] = copper_wire_df['size_max'].fillna(copper_wire_df['size_max'].median())
    copper_wire_df['vel_min'] = copper_wire_df['vel_min'].fillna(copper_wire_df['vel_min'].median())
    copper_wire_df['vel_mean'] = copper_wire_df['vel_mean'].fillna(copper_wire_df['vel_mean'].median())
    copper_wire_df['vel_max'] = copper_wire_df['vel_max'].fillna(copper_wire_df['vel_max'].median())
    copper_wire_df['part_no'] = copper_wire_df['part_no'].fillna(copper_wire_df['part_no'].median())
    # copper_wire_df['quality'] = copper_wire_df['quality'].fillna(copper_wire_df['quality'].median())
    copper_wire_df['level_o2'] = copper_wire_df['level_o2'].fillna(copper_wire_df['level_o2'].median())
    copper_wire_df['temp'] = copper_wire_df['temp'].fillna(copper_wire_df['temp'].median())
    copper_wire_df['qbin'] = copper_wire_df['qbin'].fillna(copper_wire_df['qbin'].median())


def RemoveNanAndInfWithMode(copper_wire_df):
    copper_wire_df['sum_eddy_f1'] = copper_wire_df['sum_eddy_f1'].fillna(copper_wire_df['sum_eddy_f1'].mode()[0])
    copper_wire_df['sum_eddy_f2'] = copper_wire_df['sum_eddy_f2'].fillna(copper_wire_df['sum_eddy_f2'].mode()[0])
    copper_wire_df['sum_eddy_f3'] = copper_wire_df['sum_eddy_f3'].fillna(copper_wire_df['sum_eddy_f3'].mode()[0])
    copper_wire_df['sum_ferro_f1'] = copper_wire_df['sum_ferro_f1'].fillna(copper_wire_df['sum_ferro_f1'].mode()[0])
    copper_wire_df['sum_ferro_f2'] = copper_wire_df['sum_ferro_f2'].fillna(copper_wire_df['sum_ferro_f2'].mode()[0])
    copper_wire_df['sum_ferro_f3'] = copper_wire_df['sum_ferro_f3'].fillna(copper_wire_df['sum_ferro_f3'].mode()[0])
    copper_wire_df['size_min'] = copper_wire_df['size_min'].fillna(copper_wire_df['size_min'].mode()[0])
    copper_wire_df['size_max'] = copper_wire_df['size_max'].fillna(copper_wire_df['size_max'].mode()[0])
    copper_wire_df['vel_min'] = copper_wire_df['vel_min'].fillna(copper_wire_df['vel_min'].mode()[0])
    copper_wire_df['vel_mean'] = copper_wire_df['vel_mean'].fillna(copper_wire_df['vel_mean'].mode()[0])
    copper_wire_df['vel_max'] = copper_wire_df['vel_max'].fillna(copper_wire_df['vel_max'].mode()[0])
    copper_wire_df['part_no'] = copper_wire_df['part_no'].fillna(copper_wire_df['part_no'].mode()[0])
    # copper_wire_df['quality'] = copper_wire_df['quality'].fillna(copper_wire_df['quality'].mode()[0])
    copper_wire_df['level_o2'] = copper_wire_df['level_o2'].fillna(copper_wire_df['level_o2'].mode()[0])
    copper_wire_df['temp'] = copper_wire_df['temp'].fillna(copper_wire_df['temp'].mode()[0])
    copper_wire_df['qbin'] = copper_wire_df['qbin'].fillna(copper_wire_df['qbin'].mode()[0])

    def standardise_data(X_train, X_test):
        # Initialise a new scaling object for normalising input data
        sc = StandardScaler()

        # Set up the scaler just on the training set
        sc.fit(X_train)

        # Apply the scaler to the training and test sets
        train_std = sc.transform(X_train)
        test_std = sc.transform(X_test)

        return train_std, test_std

if __name__ == '__main__':
    run_classifier()
