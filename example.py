import numpy as np
import DBSCAN_GPU as gpu
from sklearn import datasets
import matplotlib.pyplot as plt
import time
import torch

iris = datasets.load_iris()
data = np.asarray(iris.data[:, :], dtype=np.float32)

torch.cuda.synchronize()

t0 = time.time()
C = gpu.GPU_DBSCAN(data, eps=.25, minPts=5)
t1 = time.time()

print(t1-t0, C)

plt.scatter(data[:, 0], data[:, 1], c=C)
plt.show()

t0 = time.time()
C = gpu.G_DBSCAN(data, eps=.25, minPts=5)
t1 = time.time()

print(t1-t0, C)

plt.scatter(data[:, 0], data[:, 1], c=C)
plt.show()