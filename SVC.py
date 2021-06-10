import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


import pandas as pd
import numpy as np
import os
import pyreadstat
import seaborn as sns
import matplotlib.pyplot as plt
from sas7bdat import SAS7BDAT

#notes
#Coersion

def run_classifier():
    ## Load data
    copper_wire_df, meta = pyreadstat.read_sas7bdat('copper_wire_bin.sas7bdat')
    copper_wire_df.drop('quality', inplace=True, axis=1)
    ## Data preprocessing
    # remove rows with negative values
    copper_wire_df = copper_wire_df[copper_wire_df.select_dtypes(include=[np.number]).ge(0).all(1)]

    # Remove Nan and Infinite values (Choose one option)
    # RemoveNanAndInfWithDeletimgRecords(copper_wire_df)
    # RemoveNanAndInfWithMedian(copper_wire_df)

    # fill NaN and infinite values with the most frequently value occured //best solution
    RemoveNanAndInfWithMode(copper_wire_df)

    ## Separating predictors and response
    y = copper_wire_df.iloc[:, 14]
    X = copper_wire_df.iloc[:, :14]

    ## Feature selection
    ##univariate feature selection
    # @source https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
    # between 2 and 3 huge difference
    # Wyniki od 5 do 6  ficzerów ma realny wpływ, z czego dwa ostatnie są 4.5k i 3,3k, gdzie największe maja 20 i 16k
    # X = SelectKBest(score_func=chi2, k=13).fit_transform(X, y)

    ##PCA as features
    # X = perform_pca(X, copper_wire_df)

    ## Split for training and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    ##undersample // nice results
    # X_train, y_train = undersample(X_train, y_train)

    ##oversample // słabiutko
    # X_train, y_train = oversample(X_train, y_train)

    ## Classifier training using SVM
    # @param
    # C= (denotes penelty), gamma can be == to 'scale'
    svc_classifier = SVC(kernel='linear',
                         gamma='auto')  # kernel = 'Linear'  lub inne Polynominal, Radial basis function, Sigmoid,

    svc_classifier.fit(X_train, y_train)
    y_predict_svc = svc_classifier.predict(X_test)

    ## Results - accuracy
    print("SVC:")
    print("Accuracy:", accuracy_score(y_test, y_predict_svc))
    print(classification_report(y_test, y_predict_svc))
    print("confusion matrix:")
    print(confusion_matrix(y_test, y_predict_svc))


def perform_pca(X, copper_wire_df):
    scaler = StandardScaler()
    scaler.fit(copper_wire_df)
    scaled_data = scaler.transform(copper_wire_df)
    pca = PCA(n_components=7)
    pca.fit(scaled_data)
    X = pca.transform(scaled_data)
    print("shapes before PCA and after the PCA")
    print(scaled_data.shape)
    print(X.shape)
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


if __name__ == '__main__':
    run_classifier()
