# Import Libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data by using the read_csv method on the Pandas library and passing
# it the location of our data.
dataset = pd.read_csv('boston.csv')

# The next step is to look at what our data contains and it shape
print('\nFirst 5 entries\n')
print(dataset.head(5))
print()
total_row, total_col = dataset.shape
print('Total rows    : ', total_row)
print('Total columns : ', total_col)
print()

# Filter for use
x = dataset.drop(['Unnamed: 0', 'medv'], axis=1)
y = dataset['medv']

# To split our dataset into train and test splits
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# importing the model
regressor = LinearRegression()

# Regression
regressor.fit(x_train, y_train)

# To predict and to evaluate our model
y_pred = regressor.predict(x_test)
mse = mean_squared_error(y_test, y_pred)

# Finally, we plot a graph of our output to get an idea of the distribution.
plt.scatter(y_test, y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.savefig('plot1.png', format='png')
