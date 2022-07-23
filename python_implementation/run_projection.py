from python_implementation import utils
from python_implementation.Projector import Projector
from python_implementation.Constraints import Constraints
from python_implementation import Evaluator
from scipy import io
from python_implementation.utils import interpolate
import torch
from matplotlib import pyplot as plt
import time


#DATAPATH = 'citiesTSPexample.mat'
#Scanner Params([Lustig et al, IEEE TMI 2008])
def proj_handler(s0,num_iters, alpha = 3.4, beta = 0.17,disp=False):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Scanner Params([Lustig et al, IEEE TMI 2008])
    Gmax = 40e-3  # T/m
    Smax = 150e-3  # T/m/ms
    Kmax = 600  # m^-1

    gamma = 42.576 * 1e3  # kHz/T

    alpha = gamma * Gmax
    beta = gamma * Smax
    dt = 0.004  # sampling time in ms

    #data = torch.Tensor(io.loadmat(DATAPATH)['pts']).to(device) * Kmax
    start = time.time()


    multitraj = False

    if len(s0.shape) < 4: #multitraj
        multitraj = True
        s0 = s0.unsqueeze(0)


    if disp:
        # plot initial interpolated trajectory (before projection)
        plt.style.use('seaborn')
        fig, ax = plt.subplots()
        ax.plot(s0[:, 0].to('cpu'), s0[:, 1].to('cpu'), color='#000000')
        ax.set_title('Before Projection')
        plt.tight_layout()
        plt.show()

    proj = Projector(num_iters=num_iters, device=device, display_res=disp, eps_inf=0, eps2=0)
    kc = [Constraints(Evaluator.LInf2Norm(), alpha, dt, 0), Constraints(Evaluator.LInf2Norm(), beta * dt, dt, 1)]
    proj.setKinematic(kc)

    s1 = torch.clone(s0)
    for i in range(s0.shape[0]):
        s1[i] = proj(s0[i])
    end = time.time()
    if disp:
        print(f'runtime: {end - start}')
    if multitraj:  # multitraj
        s1 = s1.squeeze(0)
    return s1
