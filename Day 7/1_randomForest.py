'''
A Random Forest is a model that uses many trees and makes predictions by using averages of predictions from each tree.
It is usually more accurate than a single decision tree.
'''
import pandas as pd

path = 'data/melb_data.csv'
data = pd.read_csv(path)
data = data.dropna(axis=0)

y = data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = data[melbourne_features]

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
preds = forest_model.predict(val_X)

print(mean_absolute_error(val_y, preds))

