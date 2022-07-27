import torch
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

    def setKinematic(self, kc):
        self.kc = kc

    def _run_alg(self, s0, kc=None):
        '''s0 - initial curve to project
           kc - kinematic constraints instance. If not passed, self.kc is taken. If inexistent - error.
        '''
        if kc is None and not hasattr(self, 'kc'):
            raise ValueError("No kinematic constraint instance found as attr or param")
        elif kc is None:
            kc = self.kc
        noc = len(kc)  # num of constraints
        sensor_num = s0.shape[0]
        # Compute initial distance to constraints
        if self.disp:
            single_sensor_d = torch.zeros((noc, self.nit)).to(self.device)
            # for expanded first dim
            d = single_sensor_d.repeat(sensor_num, 1).view(-1, *(single_sensor_d.shape))
            for i in range(0, noc):
                d[:, i, 0] = torch.max(kc[i].eval.norm(kc[i].grad_operator(s0)) - kc[i].bound)

        # As being the matrix encapsulating the kinematic constraints, pre-calculate As*S (S being the curve)
        As = torch.cat((kc[0].grad_operator(s0), kc[1].grad_operator(s0))).reshape(*s0.shape, -1).to(self.device)
        Q = torch.zeros_like(As)
        R = Q.clone()

        ATq_sum_last = torch.zeros_like(s0)

        # Algorithm Core
        for k in range(0, self.nit):

            '''compute A(i)*s(i) for each constraint - used in gradient calculation later'''
            if k > 0:
                ATq_sum = ATq_sum_last
            else:
                ATq_sum = ([kc[0].Trans_operator(Q[:, :, 0], kc[0].dt) + kc[1].Trans_operator(Q[:, :, 1], kc[1].dt)])[
                    0].to(self.device)

            if self.disp:
                CF = torch.zeros((sensor_num, self.nit, 1)).to(self.device)
                for i in range(0, noc):
                    CF[:, k] = CF[:, k] - kc[i].bound * kc[i].eval.dual(Q[:, :, i])
                CF[:, k] = CF[:, k] - (0.5 * torch.norm((torch.norm(ATq_sum.view(sensor_num, -1), 2, dim=1)). \
                                                        view(sensor_num, -1), 2, dim=1) ** 2 + torch.sum(
                    (s0.view((sensor_num, -1))) * \
                    ATq_sum.view(sensor_num, -1), dim=1)).view(-1, 1)

            # computation for next iteration
            R_prev = torch.clone(R)
            z = s0 - ATq_sum
            s_star = z

            R[:, :, :, 0] = kc[0].eval.prox(Q[:, :, 0] + (1 / self.L) * (kc[0].grad_operator(s_star)), \
                                            (1 / self.L) * kc[0].bound)

            R[:, :, :, 1] = kc[1].eval.prox(Q[:, :, 1] + (1 / self.L) * (kc[1].grad_operator(s_star)), \
                                            (1 / self.L) * kc[1].bound)
            Q = R + (k) / (k + 1) * (R - R_prev)

            # ATq_sum = torch.zeros_like(s0)
            # for i in range(0,noc):
            #     ATq_sum = ATq_sum + kc[i].Trans_operator(Q[:,:, i])

            ATq_sum = kc[0].Trans_operator(Q[:, :, 0]) + kc[1].Trans_operator(Q[:, :, 1])

            if torch.max(torch.norm((ATq_sum - ATq_sum_last).view(sensor_num, -1), 2, dim=1)).item() < self.eps2 and \
                    torch.max(torch.norm((ATq_sum - ATq_sum_last).view(sensor_num, -1), \
                                         float('inf'), dim=1)).item() < self.eps_inf:
                break

            ATq_sum_last = ATq_sum

            # new dist to constraints
            if self.disp:
                for i in range(0, noc):
                    d[:, i, k] = torch.nn.ReLU()(kc[i].eval.norm(kc[i].grad_operator(s_star)) - kc[i].bound)
        # Compute output. Break if there's no change in output

        s = s0 - ATq_sum

        # display results
        if self.disp:
            s0 = s0.to('cpu')
            s = s.to('cpu')
            colors = ['#F10000', '#0C00F1']
            if self.disp:
                plt.style.use('seaborn')
                # cons = torch.zeros(s.shape[0], noc)
                fig, ax = plt.subplots(nrows=noc, ncols=sensor_num)
                for i in range(0, noc):
                    for j in range(0, sensor_num):
                        # cons[:, i]=kc[i].eval.f_space(kc[i].grad_operator(s))
                        ax[i, j].plot(s0[j, :, 0], s0[j, :, 1], color='#000000', label='orig')
                        # ax[i].plot(s, cons[:,i], color=colors[i%len(colors)], label='projected')
                        ax[i, j].scatter(s[j, :, 0], s[j, :, 1], color=colors[i % len(colors)], linestyle='--',
                                         label='projected', s=5)

                        ax[i, j].legend()
                        ax[i, j].set_title(f'Constraint {i + 1} Sensor {j + 1}')
                # plt.tight_layout()
                plt.show()
        return s.to(self.device)

    def __call__(self, curve, kc=None):
        return self._run_alg(curve, kc)










