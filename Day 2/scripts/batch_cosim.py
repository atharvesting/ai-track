import numpy as np

M = np.random.randn(4, 3)
q = np.random.randn(3)

arr1 = np.array(M)
arr2 = np.array(q)

# Normalization

arr1_l2_norm = np.linalg.norm(arr1, axis=1, keepdims=True)  # Specifying axis is necessary for row-wise normalisation.
arr1 /= arr1_l2_norm  # Correct use of broadcasting

arr2_l2_norm = np.linalg.norm(arr2)
arr2 /= arr2_l2_norm

print(arr1)
print()
print(arr2)
print()

dot_arr1_arr2 = arr1 @ arr2  # Since L2 norms are equal to 1, dot product is equal to the cosine similarity

print(dot_arr1_arr2)