import matplotlib.pyplot as plt
import numpy as np

# AI-aided

fig, axs = plt.subplots(2, 2)  # Creates a figure and 2 rows, 2 columns inside it
x = np.linspace(0, 2*np.pi, 20)  # Creates an np array with start 0, end 2*pi and 20 equally-spaced steps in between

axs[0,0].plot(x, np.sin(x))  # Top-left subplot
axs[0,1].scatter(x, np.cos(x))  # Top-right
axs[1,0].plot(x, np.cos(x))  # Bottom-left
axs[1,1].hist(np.cos(x))  # Bottom-right

plt.show()