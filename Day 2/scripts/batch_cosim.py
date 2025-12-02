import numpy as np

M = np.random.randn(4, 3)
q = np.random.randn(3)


arr1 = np.array(M)
arr2 = np.array(q)

print(f"Shape of arr1 = {np.shape(arr1)} and Shape of arr2 = {np.shape(arr2)}")

arr2 = arr2[:, np.newaxis]

print(f"Shape of arr1 = {np.shape(arr1)} and New Shape of arr2 = {np.shape(arr2)}")

"""
These matrices are still not compatible for broadcasting because neither dimension is 1 or equal.
For two matrices with shapes (4,3) and (3,1):
    - 4 and 3 are not compatible because neither is 1 or equal.
    - 3 and 1 are compatible because one of the values is 1
Thus, reshaping is necessary to make them broadcasting compatible.

"""

arr2 = arr2.reshape((1,3))

print(f"Shape of arr1 = {np.shape(arr1)} and Latest Shape of arr2 = {np.shape(arr2)}")

print()
print("Now, the two arrays can be broadcasted one over the other -\n")

def cosim(v1, v2):
    v1_l2_norm = np.linalg.norm(v1)
    v2_l2_norm = np.linalg.norm(v2)

    dot_v1_v2 = np.dot(v1, v2[0])
    cosim_v1_v2 = dot_v1_v2 / (v1_l2_norm * v2_l2_norm)
    
    return cosim_v1_v2

for i in range(4):
    print(f"Cosine similarity of {arr1[i]} and {arr2} =")
    print(f"{cosim(arr1[i], arr2):.2f}")