import numpy as np
import math as mt

v1 = np.array([-1, 2])
v2 = np.array([1, 2])

mag1 = np.linalg.norm(v1)
mag2 = np.linalg.norm(v2)

print(f"Magnitude or L2 Norm of v1 = {mag1:.2f} units")
print(f"Magnitude or L2 Norm of v2 = {mag2:.2f} units")

dot_v1_v2 = np.dot(v1, v2)

print(f"Dot product of v1 and v2 = {dot_v1_v2}")


"""
The Cosine Similarity:

- The measure of similarity between two non-zero vectors by determining the cosine of the angle they form.
- Only on the basis of direction, not magnitude.
- Like a cosine function, the score range is from -1 to 1
    - 1 = Perfect similarity
    - 0 = Orthogonality (no similarity)
    - (-1) = Perfect dissimilarity (polar opposites)
    
- Formula: cosim ≡ dot(v, w) / (||v|| * ||w||)

"""

cosim_v1_v2 = dot_v1_v2 / (mag1 * mag2)
angle_v1_v2 = mt.acos(cosim_v1_v2)

print(f"Angle formed = {angle_v1_v2:.2f} and Cosine similarity = {cosim_v1_v2:.2f}")