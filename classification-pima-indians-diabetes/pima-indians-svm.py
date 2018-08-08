# Import libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Read dataset
dataset = pd.read_csv('diabetes.csv')

# Separate into features and labels
features = dataset.drop(['Outcome'], axis=1)
labels = dataset['Outcome']

# Split into training data and test data
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.25)

# Make the classifier
classifier = SVC(kernel='linear')
classifier.fit(features_train, labels_train)

# PREDICT!!
print('\nPredicting...')
pred = classifier.predict(features_test)
print('Done!')

# Determine accuracy
accuracy = accuracy_score(labels_test, pred)

print('\nAccuracy : {}\n'.format(accuracy))
