import numpy as np
import pandas as pd

data = {
    'id': [1,2,3,4,5],
    'name': ['hello', 'there', 'i', 'am', 'python'],
    'description': ['5 letters', '5 letters', '1 letters', '2 letters', '6 letters']
}

df = pd.DataFrame(data)

M = np.random.randn(5,3)
q = np.random.randn(3)

M /= np.linalg.norm(M, axis=1, keepdims=True)
q /= np.linalg.norm(q)

similarity = M @ q

df['Similarity'] = similarity

print(df)



