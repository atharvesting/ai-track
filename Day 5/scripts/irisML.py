import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
# from urllib.request import urlretrieve

# iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# urlretrieve(iris)

df = pd.read_csv('Day 5/iris.csv')
# attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
# df.columns = attributes

'''
The original iris dataset was sorted on the basis of class.
To ensure that the dataset splits have a proper spread of entries,
The dataset rows are shuffled and then the indices are reset.

This ensures that training can encompass an equal distribution of cases.

'''
# df = df.sample(frac=1).reset_index(drop=True)

# df.to_csv('Day 4/iris.csv', index=False)
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = df[features]
y = df['class']

train_X , val_X, train_y, val_y = train_test_split(X, y, random_state=1)

parameter_grid = {"max_leaf_nodes": range(2, 50)}
search = GridSearchCV(
    DecisionTreeClassifier(random_state=1),
    param_grid=parameter_grid,
    scoring="accuracy",
    cv = 10,
    return_train_score=True
)

search.fit(train_X, train_y)

best_model = search.best_estimator_
predictions = best_model.predict(val_X)
# print(search.best_params_["max_leaf_nodes"])
# print(accuracy_score(val_y, predictions))
analysis = pd.DataFrame(search.cv_results_)
# print(analysis.columns)

fig, ax = plt.subplots()

x = analysis['param_max_leaf_nodes']
y1 = analysis['mean_train_score']
y2 = analysis['mean_test_score']
lower = analysis['mean_test_score'] - analysis['std_test_score']
upper = analysis['mean_test_score'] + analysis['std_test_score']

ax.plot(x, y1, label='Train')
ax.plot(x, y2, label='Test')
ax.set_title('Model performance visualisation')
ax.set_xlabel('Maximum number of leaf nodes')
ax.set_ylabel('Accuracy')

ax.fill_between(
    analysis['param_max_leaf_nodes'],  # x-values
    lower,                             # y-bottom  
    upper,                             # y-top
    alpha=0.3, color='red'             # transparency + color
)

ax.legend()
plt.show()


'''
These rudimentary functions can be replaced by the GridSearchCV function:

def get_acc_score(max_node_leafs, train_X, val_X, train_y, val_y):
    iris_model = DecisionTreeClassifier(max_leaf_nodes=max_node_leafs, random_state=0)
    iris_model.fit(train_X, train_y)
    predictions = iris_model.predict(val_X)
    acc_score = accuracy_score(val_y, predictions)
    return acc_score
    
def optimal_mnl(start, end):
    best = -1
    for max_node_leafs in range(start, end):
        mae = get_acc_score(max_node_leafs, train_X, val_X, train_y, val_y)
        if mae > best:
            best = mae
            optimal = max_node_leafs
            
    return optimal
'''
        
    


