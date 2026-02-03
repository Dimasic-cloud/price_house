import pandas as pd


# read Housing.csv
df = pd.read_csv('Housing.csv')

# list to fix some of features and fix this feature +1 new feature
fix_feature = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[fix_feature] = df[fix_feature].replace({'yes': 1, 'no': 0})
df['furnishingstatus'] = df['furnishingstatus'].replace({'furnished': 1, 'unfurnished': 0})
