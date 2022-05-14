import torch
from torch import nn
from models.rec_models.unet_model import UnetModel
import data.transforms as transforms
from pytorch_nufft.nufft import nufft, nufft_adjoint
import numpy as np
import matplotlib.pylab as P


class Subsampling_Layer(nn.Module):
    def initilaize_trajectory(self,trajectory_learning,initialization, n_shots):
        # x = torch.zeros(self.num_measurements, 2)
        sampel_per_shot = 3001
        sampel_per_shot = 513
        if initialization == 'spiral':
            x = np.load(f'spiral/{n_shots}int_spiral_low.npy')
            x = torch.tensor(x[:, :sampel_per_shot, :]).float()
        elif initialization == 'spiral_high':
            x = np.load(f'spiral/{n_shots}int_spiral.npy') * 10
            x = torch.tensor(x[:, :sampel_per_shot, :]).float()
        elif initialization == 'EPI':
            x = torch.zeros(n_shots, sampel_per_shot, 2)
            v_space = self.res // n_shots
            for i in range(n_shots):
                index = 0
                for j in range(sampel_per_shot):
                    x[i, index, 1] = (i + 0.5) * v_space - 160
                    x[i, index, 0] = j * 320 / sampel_per_shot - 160
                    index += 1
        elif initialization == 'cartesian':
            x = torch.zeros(n_shots, sampel_per_shot, 2)
            if n_shots < 8:
                raise ValueError
            y_list = [-4,-1,-2,-1,0,1,2,3]
            if n_shots > 8:
                y_space = int(155 // ((n_shots - 8)/2))
                y_list = y_list + list(range(4 + y_space, 160, y_space)) + [-i for i in range(4 + y_space, 160, y_space)]
            for i, y in enumerate(y_list):
                index = 0
                for j in range(sampel_per_shot):
                    x[i, index, 1] = y
                    x[i, index, 0] = j * 320 / sampel_per_shot - 160
                    index += 1
        elif initialization == 'radial':
            x = torch.zeros(n_shots, sampel_per_shot, 2)
            theta = np.pi / n_shots
            for i in range(n_shots):
                L = torch.arange(-160, 160, 320 / sampel_per_shot).float()
                x[i, :, 0] = L * np.cos(theta * i)
                x[i, :, 1] = L * np.sin(theta * i)
        elif initialization == 'uniform':
            x = (torch.rand(n_shots, sampel_per_shot, 2) - 0.5) * self.res
        elif initialization == 'gaussian':
            x = torch.randn(n_shots, sampel_per_shot, 2) * self.res/6
        else:
            print('Wrong initialization')
        self.x = torch.nn.Parameter(x, requires_grad=bool(int(trajectory_learning)))
        return

    def initilaize_trajectories(self,trajectory_learning,initialization, n_shots, num_trajectories):
        # x = torch.zeros(self.num_measurements, 2)
        sampel_per_shot = 3001
        sampel_per_shot = 513
        if initialization == 'spiral':
            x = np.load(f'spiral/{n_shots}int_spiral_low.npy')
            x = torch.tensor(x[:, :sampel_per_shot, :]).float()
        elif initialization == 'spiral_high':
            x = np.load(f'spiral/{n_shots}int_spiral.npy') * 10
            x = torch.tensor(x[:, :sampel_per_shot, :]).float()
        elif initialization == 'EPI':
            x = torch.zeros(n_shots, sampel_per_shot, 2)
            v_space = self.res // n_shots
            for i in range(n_shots):
                index = 0
                for j in range(sampel_per_shot):
                    x[i, index, 1] = (i + 0.5) * v_space - 160
                    x[i, index, 0] = j * 320 / sampel_per_shot - 160
                    index += 1
        elif initialization == 'cartesian':
            x = torch.zeros(n_shots, sampel_per_shot, 2)
            if n_shots < 8:
                raise ValueError
            y_list = [-4,-1,-2,-1,0,1,2,3]
            if n_shots > 8:
                y_space = int(155 // ((n_shots - 8)/2))
                y_list = y_list + list(range(4 + y_space, 160, y_space)) + [-i for i in range(4 + y_space, 160, y_space)]
            for i, y in enumerate(y_list):
                index = 0
                for j in range(sampel_per_shot):
                    x[i, index, 1] = y
                    x[i, index, 0] = j * 320 / sampel_per_shot - 160
                    index += 1
        elif initialization == 'radial':
            x = torch.zeros(n_shots, sampel_per_shot, 2)
            theta = np.pi / n_shots
            for i in range(n_shots):
                L = torch.arange(-160, 160, 320 / sampel_per_shot).float()
                x[i, :, 0] = L * np.cos(theta * i)
                x[i, :, 1] = L * np.sin(theta * i)
        elif initialization == 'uniform':
            x = (torch.rand(num_trajectories, n_shots, sampel_per_shot, 2) - 0.5) * self.res
        elif initialization == 'gaussian':
            x = torch.randn(num_trajectories, n_shots, sampel_per_shot, 2) * self.res/6
        else:
            print('Wrong initialization')
        
        if initialization != 'uniform' and  initialization != 'gaussian' and num_trajectories is not None:
            x = x.squeeze(0).repeat(num_trajectories, 1, 1, 1)

        self.x = torch.nn.Parameter(x, requires_grad=bool(int(trajectory_learning)))
        return

    def __init__(self, decimation_rate, res,trajectory_learning,initialization,n_shots,interp_gap,SNR=False, device='cuda', num_trajectories=None):
        super().__init__()

        self.decimation_rate=decimation_rate
        self.res=res
        self.num_measurements=res**2//decimation_rate
        if num_trajectories is not None:
            self.initilaize_trajectories(trajectory_learning, initialization, n_shots, num_trajectories=num_trajectories)
        else:
            self.initilaize_trajectory(trajectory_learning, initialization, n_shots)
        self.SNR=SNR
        self.interp_gap = interp_gap
        self.device = device

    def forward(self, input):
        # interpolate
        if self.interp_gap > 1:
            assert(len(self.x.shape) == 3 or len(self.x.shape) == 4)
            if len(self.x.shape) == 3:
                t = torch.arange(0, self.x.shape[1], device=self.x.device).float()
                t1 = t[::self.interp_gap]
                x_short = self.x[:, ::self.interp_gap, :]
                for shot in range(x_short.shape[0]):
                    for d in range(2):
                        self.x.data[shot, :, d] = self.interp(t1, x_short[shot, :, d], t)
                x_full = self.x.reshape(-1, 2)
                input = input.permute(0, 1, 4, 2, 3)
                sub_ksp = nufft(input, x_full, device=self.device)
                if self.SNR:
                    noise_amp=0.01
                    noise = noise_amp * torch.randn(sub_ksp.shape)
                    sub_ksp = sub_ksp + noise.to(sub_ksp.device)
                output = nufft_adjoint(sub_ksp, x_full, input.shape, device=self.device)

            elif len(self.x.shape) == 4:
                t = torch.arange(0, self.x.shape[2], device=self.x.device).float()
                t1 = t[::self.interp_gap]
                x_short = self.x[:, :, ::self.interp_gap, :]
                for frame in range(x_short.shape[0]): 
                    for shot in range(x_short.shape[1]):
                        for d in range(2):
                            self.x.data[frame, shot, :, d] = self.interp(t1, x_short[frame, shot, :, d], t)
                output = []
                for frame in range(x_short.shape[0]): 
                    x_full = self.x[frame].reshape(-1, 2)
                    curr_input = input[:,frame].permute(0, 3, 1, 2)
                    sub_ksp = nufft(curr_input.unsqueeze(1), x_full, device=self.device)
                    if self.SNR:
                        noise_amp=0.01
                        noise = noise_amp * torch.randn(sub_ksp.shape)
                        sub_ksp = sub_ksp + noise.to(sub_ksp.device)
                    output.append(nufft_adjoint(sub_ksp, x_full, curr_input.unsqueeze(1).shape, device=self.device))
                output = torch.cat(output, dim=1)

        return output.permute(0, 1, 3, 4, 2)

    def get_trajectory(self):
        return self.x

    def h_poly(self, t):
        tt = [ None for _ in range(4) ]
        tt[0] = 1
        for i in range(1, 4):
            tt[i] = tt[i-1]*t
        A = torch.tensor([
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1]
        ], dtype=tt[-1].dtype)
        return [
            sum(A[i, j] * tt[j] for j in range(4))
            for i in range(4)]

    def interp(self, x, y, xs):
      m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
      m = torch.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
      I = P.searchsorted(x[1:].detach().cpu(), xs.detach().cpu())
      dx = (x[I+1]-x[I])
      hh = self.h_poly((xs-x[I])/dx)
      return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx

    def __repr__(self):
        return f'Subsampling_Layer'

class Subsampling_Model(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob,decimation_rate,res,
                 trajectory_learning,initialization,n_shots, interp_gap, multiple_trajectories=False, SNR=False, device='cuda'):
        super().__init__()
        self.device = device
        if multiple_trajectories:
            self.subsampling=Subsampling_Layer(decimation_rate, res,trajectory_learning,initialization, n_shots,interp_gap, SNR, device=device, num_trajectories=in_chans)
        else:
            self.subsampling=Subsampling_Layer(decimation_rate, res,trajectory_learning,initialization, n_shots,interp_gap, SNR, device=device)
        self.reconstruction_model = UnetModel(in_chans, out_chans, chans, num_pool_layers, drop_prob)

    def forward(self, input):
        input = self.subsampling(input)
        input = transforms.complex_abs(input)
        output = self.reconstruction_model(input)
        return output

    def get_trajectory(self):
        return self.subsampling.get_trajectory()
