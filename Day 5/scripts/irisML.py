import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# from urllib.request import urlretrieve

# iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# urlretrieve(iris)

df = pd.read_csv('Day 5/iris.csv')
# attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
# df.columns = attributes

df = df.sample(frac=1).reset_index(drop=True)

# df.to_csv('Day 4/iris.csv', index=False)
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = df[features]
y = df['class']

train_X , val_X, train_y, val_y = train_test_split(X, y, random_state=1)

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

mnl = optimal_mnl(2, 5000)
print(mnl)
    
        
    


