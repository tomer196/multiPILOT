import torch
import utils
from matplotlib import pyplot as plt

class Projector:
    def __init__(self, num_iters, device, lipschitz_const=16, dstep=0.004, eps2=5e-1, eps_inf=10e-2, \
                 display_res=True):
        ''' num_iters - number of iterations the algorithm would run
            lipshcitz_const - discretization slope upperbound
            dstep - discretization step size
            eps2 - l2 distance between iteration outputs to stop iterating
            eps_inf - l_inf distance between iteration outputs to stop iterating
            display_res - to view curve and projection (could be heavier computationally)
        '''
        self.nit = int(num_iters)
        self.device = device
        self.L = lipschitz_const
        self.dstep = dstep
        self.eps2 = eps2
        self.eps_inf = eps_inf
        self.disp = display_res

    def setKinematic(self,kc):
        self.kc = kc

    def _run_alg(self, s0, kc=None):
        '''s0 - initial curve to project
           kc - kinematic constraints instance. If not passed, self.kc is taken. If inexistent - error.
        '''
        if kc is None and not hasattr(self,'kc'):
            raise ValueError("No kinematic constraint instance found as attr or param")
        elif kc is None:
            kc = self.kc
        noc = len(kc) #num of constraints
        sensor_dim = s0.shape[0]
        # Compute initial distance to constraints
        if self.disp:
            d = torch.zeros((sensor_dim,noc,self.nit))
            for i in range(0,noc):
                d[:,i,0] = torch.Tensor(torch.max(kc[i].eval.norm(kc[i].grad_operator(s0)) - kc[i].bound))

        #As being the matrix encapsulating the kinematic constraints, pre-calculate As*S (S being the curve)
        As = torch.zeros(tuple((*(s0.shape),noc)))
        for i in range(0,noc):
            As[:,:, i]=kc[i].grad_operator(s0)
        Q = torch.zeros(As.shape)
        R = Q.clone()


        ATq_sum_last = 0
        for k in range(0,self.nit):

            '''compute A(i)*s(i) for each constraint - used in gradient calculation later'''
            if k > 0:
                ATq_sum = ATq_sum_last
            else:
                ATq_sum = torch.zeros(s0.shape)
                for i in  range(0,noc):
                    ATq_sum += kc[i].Trans_operator(Q[:,:, i],kc[i].dt)

            if self.disp:
                CF = torch.zeros((self.nit, 1))
                for i in range(0,noc):
                    CF[k] = CF[k] - kc[i].bound * kc[i].eval.dual(Q[:,:, i])
                CF[k] = CF[k] - 0.5 * torch.norm(ATq_sum,2) ** 2 + torch.sum(torch.flatten(s0)*torch.flatten(ATq_sum))

            #computation for next iteration
            R_prev = torch.clone(R)
            z = s0 - ATq_sum
            s_star = z
            for i in range(0,noc):
                R[:,:, i]=kc[i].eval.prox(Q[:,:, i]+(1/self.L) * (kc[i].grad_operator(s_star)),\
                                              (1/self.L) * kc[i].bound)
            Q = R + (k) / (k + 1) * (R - R_prev)

            ATq_sum = torch.zeros(s0.shape)
            for i in range(0,noc):
                ATq_sum = ATq_sum + kc[i].Trans_operator(Q[:,:, i])
            ATq_sum_last = ATq_sum

            #new dist to constraints
            if self.disp:
                for i in range(0,noc):
                    d[i, k] = max(kc[i].eval.norm(kc[i].grad_operator(s_star)) - kc[i].bound, 0)
        #Compute output
        s = s0 - ATq_sum

        #display results
        colors = ['#F10000','#0C00F1']
        if self.disp:
            plt.style.use('seaborn')
            cons = torch.zeros(s.shape[0], noc)
            fig, ax = plt.subplots(nrows=noc,ncols=1)
            for i in range(0,noc):
                cons[:, i]=kc[i].eval.f_space(kc[i].grad_operator(s))
                ax[i].plot(s0[:,0], s0[:,1], color='#000000', label='orig')
                #ax[i].plot(s, cons[:,i], color=colors[i%len(colors)], label='projected')
                ax[i].scatter(s[:,0], s[:, 1], color=colors[i % len(colors)],linestyle='--', label='projected',s=5)

                ax[i].legend()
                ax[i].set_title(f'Constraint {i+1}')
            #plt.tight_layout()
            plt.show()

    def __call__(self, curve,kc=None):
        self._run_alg(curve,kc)










