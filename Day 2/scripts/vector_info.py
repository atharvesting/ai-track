import numpy as np

a = [1,2,3,4,5]
b = [9,8,7,6,5]

v1 = np.array(a)
v2 = np.array(b)

v1_l2_norm = np.linalg.norm(v1)
v2_l2_norm = np.linalg.norm(v2)

dot_v1_v2 = np.dot(v1, v2)
cosim_v1_v2 = dot_v1_v2 / (v1_l2_norm * v2_l2_norm)

result = f"""
For vectors {v1} and {v2}:

L2 Norm of v1 = {v1_l2_norm:.2f}
L2 Norm of v2 = {v2_l2_norm:.2f}

Dot product = {dot_v1_v2:.2f}
Cosine Similarity = {cosim_v1_v2:.2f}
"""

print(result)