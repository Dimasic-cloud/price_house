import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error


# param for GridSerchCV
param = {
    'elasticnet__alpha': [0.001, 0.01, 0.1, 1, 5, 10, 50, 100],
    'elasticnet__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1]
}

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# creating pipeline
pipe = Pipeline([
    ("standardscaler", StandardScaler()),
    ("elasticnet", ElasticNet(max_iter=10000, random_state=42))
])

# serch best param for pipeline
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param,
    scoring='neg_root_mean_squared_error',
    cv=5,
    n_jobs=-1
)

# model fitting
grid.fit(X_train, y_train)

# model predict
model = grid.best_estimator_
y_pred = model.predict(X_test)

baseline_pred = np.full_like(y_test, y_train.mean())
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse, baseline_rmse)