'''
- Model validation is a necessary to determine the predictive accuracy of a model.
- The model quality is usually summarised into a single metric.
- Mean Absolute Error (MAE) is one of these metrics used to measure accuracy.
'''
# error = abs(actual_price - predicted_price)
# Then we take the average of these values to get a value which describes -
# On average, our predictions are off by about X (kaggle)

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

file_path = 'Day 5/melb_data.csv'
m = pd.read_csv(file_path)

filtered = m.dropna(axis=0)  # Filtering out rows with missing values across the columns
y = filtered.Price
features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered[features]

model = DecisionTreeRegressor(random_state=1)
model.fit(X, y)
predictions = model.predict(X)

MAE = mean_absolute_error(y, predictions)
print("In sample score:", MAE)

'''
In-sample score:
- The MAE calculated here is an In-sample score.
- This is because the dataset used to train the model and then evaluate it is the same.
- Doing so is bad in practice.

Validation Data:
- Since models' practical value come from making predictions on new data, we measure performance on data that wasn't used to build the model. 
- The most straightforward way to do this is to exclude some data from the model-building process, and then use those to test the model's accuracy on data it hasn't seen before. 
- This data is called validation data.
(kaggle, Intro to Machine Learning)

'''

'''
In order to test our model properly, we need to test it against a dataset that it has never seen before.
The train_test_split function is used to split the dataset into training and validation set pairs.
'''
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

new_model = DecisionTreeRegressor(random_state=1)
new_model.fit(train_X, train_y)  # The model is trained only using the data specially assigned for training.
new_predictions = new_model.predict(val_X)  # The model is then tested on the validation dataset not seen before.

new_MAE = mean_absolute_error(new_predictions, val_y)  # MAE is calculated to determine model accuracy
print("True validation score:", new_MAE)