import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# sys.path.insert(0, '/home/tomerweiss/pytorch-nufft')
import matplotlib.pyplot as plt
# import cv2
import torch
import pytorch_nufft.nufft as nufft

import data.transforms as transforms
device='cpu'
# create trajectory
res=256
decimation_rate=4
dt=1e-2
num_measurements=res**2//decimation_rate
x = torch.zeros(num_measurements, 2)
# c = decimation_rate / res ** 2 * 1600
# r = torch.arange(num_measurements, dtype=torch.float64) * 1e-1
# x[:, 0] = c * r * torch.cos(r)
# x[:, 1] = c * r * torch.sin(r)
index = 0
for i in range(res // decimation_rate):
    if i % 2 == 0:
        for j in range(res):
            x[index, 1] = i * decimation_rate + decimation_rate / 2 - 160
            x[index, 0] = j - 160
            index += 1
    else:
        for j in range(res):
            x[index, 1] = i * decimation_rate + decimation_rate / 2 - 160
            x[index, 0] = res - j - 1 - 160
            index += 1

x=x.to(device).requires_grad_()

# Get data
from skimage import io
img = io.imread('DIPSourceHW1.jpg', as_gray=True).astype('float32')
img = img.reshape(1,1,256,256)
img = torch.tensor(img).to(device)
ksp=transforms.rfft2(img).unsqueeze(1).permute(0,1,3,4,2)
img = transforms.ifft2_regular(ksp).permute(0, 1, 4, 2, 3)
original_shape=img.shape

# NUFFT Forward
ksp = nufft.nufft(img, x, device=device)
# ksp=transforms.fft2(img)
# ksp=interp.bilinear_interpolate_torch_gridsample(ksp,x)

# NUFFT Adjoint
img_est = nufft.nufft_adjoint(ksp,x,original_shape,device=device)

loss=torch.nn.functional.l1_loss(img, img_est)
loss.backward()

img_est = img_est.permute(0,1,3,4,2)
plt.figure()
plt.imshow(img_est[0, 0,:,:, 0].detach().cpu().numpy(), cmap='gray')
plt.show()