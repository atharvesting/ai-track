import numpy as np
import pandas as pd

a = pd.read_csv('Day 3/data.csv')  # extract data from the CSV file into a DF

M = a[['GPA', 'ExamScore1', 'ExamScore2', 'ExamScore3']].to_numpy()  # Convert DF into n-d numpy array containing necessary numeric values

top = a.sort_values('GPA', ascending=False)
q = top[['GPA', 'ExamScore1', 'ExamScore2', 'ExamScore3']].iloc[0].to_numpy()

print(f"M shape = {np.shape(M)}, q shape = {np.shape(q)}")

mag = np.linalg.norm(M, axis=1, keepdims=True)
n = M / mag  # Using broadcasting for the normalisation operation

q /= np.linalg.norm(q)

cosim = (n @ q).flatten()  # flatten() turns an n-d array to 1-d array. Use for safety.

# Now we will attach this similarity score column to our Student DF
a['GPASimilarity'] = cosim

a.to_csv('Day 3/data.csv', index=False)


