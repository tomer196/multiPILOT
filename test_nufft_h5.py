import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# sys.path.insert(0, '/home/tomerweiss/pytorch-nufft')
import matplotlib.pyplot as plt
# import cv2
import torch
import pytorch_nufft.nufft as nufft
import pytorch_nufft.interp as interp
import h5py

import data.transforms as transforms
device='cpu'
# create trajectory
res=320
decimation_rate=10
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

# check resample all channels and then combine to 1 channel
slice = 18
with h5py.File('/ext_mnt/tomer/multicoil_val/file1000000.h5', 'r') as data:
    kspace = data['kspace'][slice]
    target = data['reconstruction_rss'][slice] if 'reconstruction_rss' in data else None

kspace = transforms.to_tensor(kspace)
image = transforms.ifft2_regular(kspace)
image = transforms.complex_center_crop(image, (320, 320)).unsqueeze(0)
image, mean, std = transforms.normalize_instance(image, eps=1e-11)
target = transforms.to_tensor(target)
target, mean, std = transforms.normalize_instance(target, eps=1e-11)
image = image.permute(0, 1, 4, 2, 3)
original_shape=image.shape

ksp = nufft.nufft(image, x, device=device)
# ksp=transforms.fft2(image.permute(0, 1, 3,4,2)).permute(0, 1, 4, 2, 3).squeeze()
# ksp=interp.bilinear_interpolate_torch_gridsample(ksp,x).unsqueeze(0)
img_est = nufft.nufft_adjoint(ksp,x,original_shape,device=device)

img_est = transforms.complex_abs(img_est.permute(0,1,3,4,2)).squeeze()
img_est = transforms.root_sum_of_squares(img_est)

plt.imsave('est_bi.png',img_est.detach().cpu().numpy(), cmap='gray')
plt.imsave('target.png',target.detach().cpu().numpy(), cmap='gray')


# check combine to 1 channel and then resample
with h5py.File('/ext_mnt/tomer/multicoil_val/file1000000.h5', 'r') as data:
    kspace = data['kspace'][slice]
    target = data['reconstruction_rss'][slice] if 'reconstruction_rss' in data else None

kspace = transforms.to_tensor(kspace)
image = transforms.ifft2_regular(kspace)
image = transforms.complex_center_crop(image, (320, 320))
image, mean, std = transforms.normalize_instance(image, eps=1e-11)
target = transforms.to_tensor(target)
target, mean, std = transforms.normalize_instance(target, eps=1e-11)

image = transforms.complex_abs(image)
image = transforms.root_sum_of_squares(image).unsqueeze(0).unsqueeze(0)
k = transforms.rfft2(image)
image = transforms.ifft2(k).unsqueeze(0)
original_shape=image.shape

# ksp = nufft.nufft(image, x, device=device)
ksp=transforms.fft2(image.permute(0, 1, 3,4,2)).permute(0, 1, 4, 2, 3).squeeze(0)
ksp=interp.bilinear_interpolate_torch_gridsample(ksp,x).unsqueeze(0)
img_est = nufft.nufft_adjoint(ksp,x,original_shape,device=device)

img_est = transforms.complex_abs(img_est.squeeze().permute(1,2,0))

plt.imsave('est_1ch_bi.png',img_est.detach().cpu().numpy(), cmap='gray')


ksp=transforms.rfft2(transforms.complex_abs(image.permute(0,1,3,4,2)))
ksp=interp.bilinear_interpolate_torch_gridsample(ksp,x)

# NUFFT Adjoint
img_est = nufft.nufft_adjoint(ksp,x,original_shape,device=device)

# loss=torch.nn.functional.l1_loss(img, img_est)
# loss.backward()

img_est = transforms.complex_abs(img_est.permute(0,1,3,4,2))
plt.figure()
plt.imshow(img_est[0, 0,:,:].detach().cpu().numpy(), cmap='gray')
plt.show()