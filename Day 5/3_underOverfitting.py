'''
(kaggle, Intro to Machine Learning)
Models can suffer from either:

Overfitting: capturing spurious patterns that won't recur in the future, leading to less accurate predictions, or
Underfitting: failing to capture relevant patterns, again leading to less accurate predictions.

Thus, fine-tuning the model is a necessary step to ensure that the model doesn't lose accuracy to under- and overfitting issues.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, train_y, val_X, val_y):
    '''
    get_mae: Calculates the MAE for a model given training data, validation data, and maximum leaf nodes allowed.
    
    :param max_leaf_nodes: The maximum depth we want to allow for the Decision Tree Model
    :param train_X: Feature subset reserved for training the model
    :param train_y: Prediction Target subset reserved for training the model
    :param val_X: Validation subset for the features
    :param val_y: Validation subset for the target prediction
    '''
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    predictions = model.predict(val_X)
    mae = mean_absolute_error(val_y, predictions)
    return mae

# Comparing the accuracy of models with different max_leaf_nodes values

data = pd.read_csv('Day 5/melb_data.csv')

filtered = data.dropna(axis=0)  # Filtering out rows with missing values across the columns
y = filtered.Price
features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

'''
In order to fine_tune the model, we need to determine the optimal value for max_leaf_nodes
so that Under- and Overfitting can be avoided.
'''
min = 5000000
for max_leaf_nodes in [5,5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, train_y, val_X, val_y)
    if my_mae < min:
        min = my_mae
    # print(f"Max leaf nodes = {max_leaf_nodes}, MAE = {my_mae}")
    
print("Optimal no. of leaf nodes allowed =", min)