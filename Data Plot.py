import matplotlib.pyplot as plt
import nnfs 
from nnfs.datasets import vertical_data,spiral_data

nnfs.init()


X1, y1 = vertical_data(samples=100, classes= 3)

plt.scatter(X1[:, 0], X1[:, 1], c=y1, s=40, cmap="brg")
plt.show()



#c

X2, y2 = spiral_data(samples=100, classes= 3)

plt.scatter(X2[:, 0], X2[:, 1], c=y2, s=40, cmap="brg")
plt.show()
