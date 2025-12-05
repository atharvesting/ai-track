import pandas as pd
from sklearn.tree import DecisionTreeRegressor

file_path = 'Day 5/melb_data.csv'

m = pd.read_csv(file_path)

m = m.dropna(axis=0)  # dropna(axis=0) removes any rows with missing values across any columns

y = m.Price  # returns a DF containing a singular feature using dot notation

features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']  # Defining the features we wish to select

X = m[features]  # Creating a new DF only containing those features

'''
Many machine learning models allow some randomness in model training. 
Specifying a number for random_state ensures you get the same results in each run. 
This is considered a good practice.
'''
model = DecisionTreeRegressor(random_state=1)  # Create a Decision Tree model
model.fit(X, y)  # Process of finding patterns in data.

print(X.head())
print(model.predict(X.head()))  # The fitted model can now make price predictions given a set of house features