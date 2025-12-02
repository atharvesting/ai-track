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