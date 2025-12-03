import pandas as pd
import numpy as np

"""
Series:
- 1-d labeled array capable of storing any datatype.
- Can be thought of as a single column in a table.

"""

s = pd.Series([1,3,5,7,9])
# print(s)  # each entry is assigned an index as shown in the output.

"""
DataFrame:
- 2-d labeled structure with columns that can hold different data types.
- Can be thought of as a table with rows and columns.

"""

# Creating a DataFrame using a Dictionary. The labels become the headers of the table and the values containing lists are the columns.
data = {
    'Word': ['Hello', 'World', 'AI'],
    'Length': [5, 5, 2],
}
f = pd.DataFrame(data)
# print(f)

# Creating a DF from a list of dictionaries.
data = [
    {'name': 'Max', 'age': 25}, 
    {'name': 'Ben', 'age': 45}, 
    {'name': 'Ten', 'age': 22}, 
    {'name': 'Gwen', 'age': 78}
]
g = pd.DataFrame(data)
print(g)

n = np.array([
    ['Hello', 5],
    ['Onomatopoeia', 12],
    ['AI', 2],
])
print()
h = pd.DataFrame(data=n, columns=['Word', 'Length'])
print(h)