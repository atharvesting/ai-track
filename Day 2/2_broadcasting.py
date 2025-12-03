import numpy as np

"""
Broadcasting

- Principle used when performing arithmetic operations with non-compatible arrays.
- Smaller array is expanded to the size of the bigger one.

- Conditions:
    - Check Dimensions: Ensure the arrays have the same number of dimensions or expandable dimensions.
    - Dimension Padding: If arrays have different numbers of dimensions the smaller array is left-padded with ones.
    - Shape Compatibility: Two dimensions are compatible if they are equal or one of them is 1.

"""

a = np.array([[2,4,65],[7,3,2],[4,5,6],[3,2,4]])
b = 10  # simple scalar

c = a + b  # a is expanded as necessary to allow for element-wise operation
print(c)

c = a * b
print(c)

d = np.array([2, 4, 6])
e = np.array([[1, 3, 5], [7, 9, 11]])
# here we have a 1x3 and a 2x3 array. a1 will be broadcasted by adding another row with the same values as row 1
f = d + e
print(f)

g = np.array(['Greater', 'Smaller'])
h = np.where(d > e, g[0], g[1])  # conditional broadcasting using where function. Syntax: np.where(condition, value_if_true, value_if_false)
print(h)

print(d[:, np.newaxis].shape)  # Reshaping from (3,) which is a 1-d array to (3,1) which is a n-d array or matrix. necessary for broadcasting if
# one of the operands is a flat list and not a vector or matrix

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

print(arr1 * arr2)