import pathlib
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
# sys.path.insert(0,'/home/tomerweiss/PILOT')

import numpy as np
import torch

from scipy.spatial import distance_matrix
from tsp_solver.greedy import solve_tsp
from common.args import Args
import matplotlib.pyplot as plt
from models.subsampling_model import Subsampling_Model
import scipy.io as sio

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = Subsampling_Model(
        in_chans=1,
        out_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob,
        decimation_rate=args.decimation_rate,
        res=args.resolution,
        trajectory_learning=args.trajectory_learning,
        initialization=args.initialization,
        SNR=args.SNR,
        n_shots=args.n_shots,
        interp_gap=args.interp_gap
    ).to(args.device)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    return model

def plot_trajectory(x,name):
    fig = plt.figure(figsize=[16,16])
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([-165, 165, -165, 165])
    ax.plot(x[:, :, 0].T, x[:, :, 1].T)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.savefig(f'{name}.png', bbox_inches='tight')
    return

def plot_trajectory_color(x,c,top, name):
    x1 = x[:, 0]
    y1 = x[:, 1]
    points = np.array([x1, y1]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=plt.get_cmap('coolwarm'))
    cc=np.maximum(np.abs(c[:,0]),np.abs(c[:,1]))
    cc[cc>top]=top
    cc[0]=top+0.01
    lc.set_array(cc)
    lc.set_linewidth(1)

    fig = plt.figure(figsize=[7, 5.5])
    axcb = fig.colorbar(lc)
    # axcb.set_label('Velocity')
    plt.gca().add_collection(lc)
    plt.axis('off')
    plt.xlim(-165, 165)
    plt.ylim(-165, 165)
    plt.show()
    fig.savefig(f'{name}.png', bbox_inches='tight')
    return

def plot_no(v,max,name):
    fig, axarr = plt.subplots(2, sharex=True)
    # plt.title('Velocity')
    axarr[0].plot(v[:,0])
    axarr[1].plot(v[:,1])

    axarr[0].set_ylim([-max*1.1, max*1.1])
    axarr[1].set_ylim([-max * 1.1, max * 1.1])

    axarr[0].get_xaxis().set_visible(False)
    axarr[1].get_xaxis().set_visible(False)
    fig.savefig(f'{name}.png', bbox_inches='tight')
    return

def plot_acc(a,a_max,name):
    a=a/0.17*200
    a_max=a_max/0.17*200
    fig, ax = plt.subplots(2, sharex=True)

    ax[0].get_xaxis().set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    # plt.title('Acceleration')
    limit=np.ones(a.shape[0])*a_max
    ax[0].plot(a[:,0])
    ax[0].plot(limit,color='red')
    ax[0].plot(-limit, color='red')
    ax[1].plot(a[:,1])
    ax[1].plot(limit, color='red')
    ax[1].plot(-limit, color='red')
    fig.savefig(f'{name}.png', bbox_inches='tight')
    return

def plot_vel(v,v_max,name):
    v=v/3.4*40
    v_max=v_max/3.4*40
    fig, ax = plt.subplots(2, sharex=True)

    ax[0].get_xaxis().set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    # plt.title('Acceleration')
    limit=np.ones(v.shape[0])*v_max

    ax[0].plot(v[:,0])
    ax[0].plot(limit, color='red')
    ax[0].plot(-limit, color='red')
    ax[1].plot(v[:,1])
    ax[1].plot(limit, color='red')
    ax[1].plot(-limit, color='red')
    fig.savefig(f'{name}.png', bbox_inches='tight')
    return

def reconstructe():
    args = create_arg_parser().parse_args(sys.argv[1:])

    print(args.checkpoint)
    model = load_model(args.checkpoint)
    x= model.module.get_trajectory()
    x=x.detach().cpu().numpy()
    # decimation_rate=80
    #
    # v = (x[1:, :] - x[:-1, :])
    # a = (v[1:, :] - v[:-1, :])

    name = 'brain_512_radial_learned_32'
    plot_trajectory(x,f'{name}_traj')
    sio.savemat(f'{name}_traj.mat',{'x':x})
    # np.save(f'{name}_traj', x)
    # a=a/0.17*40
    # plot_trajectory_color(x,a,800,f'{name}_traj')

    # plot_no(v,30,f'{name}_vel')
    # plot_no(a,3, f'{name}_acc')

    # plot_vel(v,3.4,f'{name}_vel')
    # plot_acc(a,0.17,f'{name}_acc')


def create_arg_parser():
    parser = Args()
    parser.add_argument('--checkpoint', type=pathlib.Path,default=f'/home/tomerweiss/multiPILOT2/summary/brain/32/512_radial_0.003_0.1_0.1_new/best_model.pt',
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path,default=f'rec',
                        help='Path to save the reconstructions to')
    return parser


if __name__ == '__main__':
    reconstructe()
