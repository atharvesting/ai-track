import numpy as np

a = [[2,4,65],[7,3,2],[4,5,6],[3,2,4]]  # Normal lists within list
b = np.array(a)  # b stores a numpy array object based on list a (3x4 matrix)

print(b)  # prints b in a matrix format because it is multi-dimensional
print(b.shape)  # returns a tuple (no. of rows, no. of cols)
print(b.dtype)  # returns data type of values stored in the array

c = np.reshape(b, (2, 6))  # reshapes the matrix. product of pair must equal orginal m times n
print(c.shape)

d = np.array(c, dtype=np.int16)  # creates an array with int16 (2-byte integer) data type. useful for performance
e = np.array([1.0, 2.5], dtype='f4')  # 'f4' is a shorthand for float32

print(f"Total sum = {np.sum(a, axis=None)}, Row-wise sum = {np.sum(a, axis=1)}, Column-wise sum = {np.sum(a, axis=0)}")
print(f"Mean = {np.mean(a)}, Row-wise mean = {np.mean(a, axis=1)}, Column-wise mean = {np.mean(a, axis=0)}")

f = np.array([[36,2,6,7], [3,73,13,1], [3,6,1,2]], dtype=np.int8)
print(np.dot(a,f), "\n\n",  a @ f)  # both are valid ways to find the dot products
# dot product of 1-d vectors is a scalar, whereas dot product of two n-d matrices is another matrix

g = np.array([[[4,2,8,7], [3,3,9,1], [3,1,8,6]]])
print(np.multiply(f,g), "\n\n",  f * g)  # both are valid ways to multiply arrays element-wise
print(f + g)  # element-wise addition
print(f > g)  # element-wise comparison. all relational operators can be used. returns a boolean array

h = np.ones((2,3), dtype=np.int16, order='C')  # creates an array with specified shape with all elements initialised to 1
# order='C' is used for row major order where as 'F' is used for column major order
print(h)

"""
Practice
"""

print(a[:2])