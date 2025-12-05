import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from urllib.request import urlretrieve

# iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# urlretrieve(iris)

df = pd.read_csv('Day 4/iris.csv')
# attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
# df.columns = attributes

df = df.sample(frac=1).reset_index(drop=True)

df.to_csv('Day 4/iris.csv', index=False)