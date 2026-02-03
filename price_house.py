import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# read Housing.csv
df = pd.read_csv('Housing.csv')

# list to fix some of features and fix this feature +1 new feature
fix_feature = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[fix_feature] = df[fix_feature].replace({'yes': 1, 'no': 0})
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

# data split on feature and target
target = 'price'
X = df.drop(columns=target)
y = df[target]

# X and y splitting on train and test selections
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# creating model OLS
ols = LinearRegression()
# model fitting
ols.fit(X_train, y_train)

# predict
y_pred = ols.predict(X_test)

# evaluating ols
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse, "mean:", df[target].mean())