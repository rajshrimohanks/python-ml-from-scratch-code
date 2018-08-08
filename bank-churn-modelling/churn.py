# Import libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tflearn

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# Show first five rows
print('\nObservations : \n')
print(dataset.head())

# Extract relevant data
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Make One Hot Representation of categorical features
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Split dataset to training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
y_train = np.reshape(y_train, (-1, 1))  # Reshape y_train to [None,1]
y_test = np.reshape(y_test, (-1, 1))  # Reshape y_test to [None,1]

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Build the neural net
net = tflearn.input_data(shape=[None, 11])
net = tflearn.fully_connected(net, 6, activation='relu')
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 6, activation='relu')
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 1, activation='tanh')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Train by gradient descent
model.fit(X_train, y_train, n_epoch=10, batch_size=16, validation_set=(
    X_test, y_test), show_metric=True, run_id='dense_model')
