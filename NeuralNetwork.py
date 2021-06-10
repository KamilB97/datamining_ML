import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
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
    ## Load data
    copper_wire_df, meta = pyreadstat.read_sas7bdat('copper_wire_bin.sas7bdat')
    copper_wire_df.drop('quality', inplace=True, axis=1)
    ## Remove Nan and Infinite values (Choose one option)
    # RemoveNanAndInfWithDeletimgRecords(copper_wire_df)
    # RemoveNanAndInfWithMedian(copper_wire_df)
    RemoveNanAndInfWithMode(copper_wire_df)  # best

    ## remove rows with negative values
    copper_wire_df = copper_wire_df[copper_wire_df.select_dtypes(include=[np.number]).ge(0).all(1)]

    ## Feature selection

    ## Separating predictors and response
    y = copper_wire_df.iloc[:, 14]
    X = copper_wire_df.iloc[:, :14]

    ## Split for training and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    ##undersample // nice results
    # X_train, y_train = undersample(X_train, y_train)

    ##oversample // słabiutko
    # X_train, y_train = oversample(X_train, y_train)

    ## Classifier training using Deep Learning - neural network
    # @Source https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
    nn_clasifier = Sequential()
    nn_clasifier.add(Dense(12, input_dim=14, activation='relu'))
    nn_clasifier.add(Dense(8, activation='relu'))
    nn_clasifier.add(Dense(1, activation='sigmoid'))
    nn_clasifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    nn_clasifier.fit(X_train, y_train, epochs=5,
                     batch_size=14)  # play with epochs and batch/ the lower batch the better results and loss is less
    loss, accuracy = nn_clasifier.evaluate(X_test, y_test)
    print('Accuracy: %.2f' % (accuracy * 100))
    print(
        'Loss: %.2f' % (loss * 100))  # what the loss means // I use fully connected network structure with three layers
    # they are defined using the Dense class

    # predict_train = nn_clasifier.predict(X_train)
    # predict_test = nn_clasifier.predict(X_test)

    # print(predict_test.shape)
    # print(classification_report(y_test, predict_test))


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
    # fill NaN and infinite values with the most frequently value occured
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
