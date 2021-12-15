#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
import pandas as pd
from typing import List

CAP_FEATURE_SET = ['cap-shape', 'cap-color', 'cap-surface']
STALK_FEATURE_SET = ['stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring']
GILL_FEATURE_SET = ['gill-attachment', 'gill-spacing', 'gill-size', 'gill-color']

def accuracy(confusion_matrix):
    """
    Use the confusion matrix to calculate accuracy
    """
    return confusion_matrix.trace() / confusion_matrix.sum()

def prepare_datasets(X,y,test_size):
    """
    Create dataset split and perform preprocessing
    """

    #Create the train and test datasets, with an 80:20 split
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=test_size, random_state=0)

    scaler = StandardScaler()
    scaler.fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    return (Xtrain, Xtest, ytrain, ytest)

def train_and_predict(mlp:MLPClassifier,df:pd.DataFrame,features:List):

    X = df[features]
    y = df['class']

    Xtrain, Xtest, ytrain, ytest = prepare_datasets(X,y,test_size=0.2)

    # Fit to linear regression model
    mlp.fit(Xtrain,ytrain)
    return (mlp.predict(Xtest),ytest)

def main():
    """
    Driver function
    """
    # Read in the csv data
    df = pd.read_csv("./mushrooms.csv",dtype='category')

    # Build the MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(30,30,30),max_iter=300,activation='relu',solver='adam')

    # Categories are characters - need to convert to numerical categories
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    # Create correlations heatmaps
    sns.heatmap(df[["class", *CAP_FEATURE_SET]].corr(),annot=True)
    plt.title("Cap Feature Correlations")
    plt.savefig("cap-correlations.png")
    plt.show()
    sns.heatmap(df[["class", *STALK_FEATURE_SET]].corr(),annot=True)
    plt.title("Stalk Feature Correlations")
    plt.savefig("stalk-correlations.png")
    plt.show()
    sns.heatmap(df[["class", *GILL_FEATURE_SET]].corr(),annot=True)
    plt.title("Gill Feature Correlations")
    plt.savefig("gill-correlations.png")
    plt.show()

    # Train using cap features
    y_pred,ytest = train_and_predict(mlp,df,CAP_FEATURE_SET)

    cm = confusion_matrix(y_pred,ytest)
    print("")
    print(f"Cap Feature Set Model Accuracy : %{accuracy(cm).round(2)*100}")

    # Train using gill features
    y_pred,ytest = train_and_predict(mlp,df,STALK_FEATURE_SET)

    cm = confusion_matrix(y_pred,ytest)
    print("")
    print(f"Stalk Feature Set Accuracy : %{accuracy(cm).round(2)*100}")

    # Train using veil features
    y_pred,ytest = train_and_predict(mlp,df,GILL_FEATURE_SET)

    cm = confusion_matrix(y_pred,ytest)
    print("")
    print(f"Gill Feature Set Accuracy : %{accuracy(cm).round(2)*100}")

if __name__ == "__main__":
    main()






