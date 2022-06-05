import numpy as np

a = np.array([1, 2, 3])
b = np.matrix(a)
print(a.shape)
print(b.shape)
print(b.getT().dot(b))
print(a.reshape(3, -1).dot(a.reshape(1, -1)))


