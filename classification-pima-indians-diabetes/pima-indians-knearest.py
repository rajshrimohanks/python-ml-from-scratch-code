# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load data using pandas
dataset = pd.read_csv('diabetes.csv')
print('\nFirst 5 entries\n')
print(dataset.head(5))

total_row, total_col = dataset.shape
print()
print('Total rows    : ', total_row)
print('Total columns : ', total_col)
print('\nDescription of dataset\n')
print(dataset.describe())

# Check for correlation
corr = dataset.corr()
fig, ax = plt.subplots(figsize=(13, 13))
ax.matshow(corr)
plt.title('Correlation of data')
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.savefig('correlation.png', format='png')
plt.show()

# Separate into features and labels
features = dataset.drop(['Outcome'], axis=1)
labels = dataset['Outcome']

# Split into training data and test data
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.25)

# Bring in the CLASSIFIER!
classifier = KNeighborsClassifier()
classifier.fit(features_train, labels_train)

# Make the PREDICTOR!
print('\nPredicting...')
pred = classifier.predict(features_test)
print('Done!')

# Determine accuracy
accuracy = accuracy_score(labels_test, pred)

print('\nAccuracy : {}\n'.format(accuracy))
