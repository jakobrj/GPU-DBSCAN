import numpy as np
import dbscan_gpu as gpu
import matplotlib.pyplot as plt
import time
import torch
import torchvision.datasets as torch_datasets
import umap

# loading mnist
mnist = torch_datasets.MNIST(root="data", train=True, download=True, transform=None)
data = np.asarray(mnist.data[:, :], dtype=np.float32)
data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])

reducer = umap.UMAP()
data = np.asarray(reducer.fit_transform(data), dtype=np.float32, order="c")
a = 0.1
plt.scatter(data[:, 0], data[:, 1], alpha=a)
plt.show()

eps = .5  # 2.5
minPts = 400

torch.cuda.synchronize()

t0 = time.time()
C = gpu.GPU_DBSCAN(data, eps=eps, minPts=minPts)
t1 = time.time()
print(t1 - t0, C, np.unique(C, return_counts=True))
plt.scatter(data[:, 0], data[:, 1], c=C, alpha=a)
plt.show()

t0 = time.time()
C = gpu.G_DBSCAN(data, eps=eps, minPts=minPts)
t1 = time.time()
print(t1 - t0, C, np.unique(C, return_counts=True))
plt.scatter(data[:, 0], data[:, 1], c=C, alpha=a)
plt.show()
