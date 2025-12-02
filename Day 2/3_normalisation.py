import numpy as np

a = np.array([[2,3,5,6,7], [3,4,5,2,2]])
b = np.ones((3,4))

print(np.mean(a, axis=1), a.mean(axis=1))  # two ways to get the same result

print("Standard Deviation of a =", np.std(a, axis=1))
print("Standard Deviation of b =", np.std(b, axis=1))