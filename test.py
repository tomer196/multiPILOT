import torch
import matplotlib.pyplot as plt
import numpy as np
import h5py
import matplotlib.pyplot as plt


with h5py.File('/home/tomerweiss/Datasets/singlecoil_train/file1000001.h5', 'r') as data:
    kspace = data['kspace'][18]

kspace=np.fft.fftshift(kspace)
image=np.fft.fft2(kspace)
image=np.fft.ifftshift(image)
plt.imshow(np.abs(image))
plt.show()