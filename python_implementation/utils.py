import torch

def projectOntoL1Ball(x,L,alpha):
    '''Computes the projection of x on a weighted ball defined by {x, ||L.*x||_1 \leq \alpha}.
       Returns projection z and threshold parameter sigma
    '''
    xf = x.flatten()
    Lf = L.flatten()
    n = torch.numel(x)

    #Projection if solution is trivial
    M = torch.sum(torch.abs(Lf*xf))
    if M <= alpha:
        return x,0

    if alpha <= 0:
        z = torch.zeros(x.shape)
        sigma = float('inf')
        return z,sigma

    #projection if solution is non trivial
    ax = torch.abs(xf)
    w = ax/L
    y, J = torch.sort(w)
    E1 = L[J[n-1]] * ax[J[n-1]]
    E2 = L[J[n-1]] ** 2
    Ep = 0
    E = E1 - y[n-1] * E2
    i = n-1
    while ((E < alpha) and (i > 1)):
        i = i - 1
        E1 = E1 + L[J[i]] * ax[J[i]]
        E2 = E2 + L[J[i]] ** 2
        Ep = E
        E = E1 - y[i]*E2
    if i > 0:
        a = y[i]
        b = y[i+1]
        r = (Ep - E) / (b - a)
        sigma = (alpha - (Ep - r * b)) / r
    else:
        sigma = (M - alpha) / E2
    z = xf
    K = torch.where(ax < sigma * L)[0]
    if not len(K):
        z[ax < sigma * L] = 0
    K = torch.where(ax >= sigma * L)[0].type(torch.int64)
    if not len(K):
        z[K] = x[K] - sigma * torch.sign(x[K])* L[K]
    return torch.view(z, x.shape),sigma

def firstDerivative(curve,dt=1):
    '''Calculate first order derivative of given curve with discretication step dt'''
    sp = torch.zeros(curve.shape)
    sp[:,1:,:]=curve[:,1:,:]-curve[:,:-1,:]
    return sp.to(curve.device)/dt

def primeT(curve,dt=1):
    '''calc A^T*curve, A being the derivation operator (dicrete)'''
    sp = torch.zeros(curve.shape)
    sp[:,0, :] = -curve[:,1, :]
    sp[:,1:-1, :] = -curve[:,2:, :] + curve[:,1:-1, :]
    sp[:,-1, :] = curve[:,-1, :]
    return sp.to(curve.device) / dt
def secondDerivative(curve,dt=1):
    '''Calculate 2nd order derivative of given curve with discretication step dt'''
    first_der = firstDerivative(curve,dt)
    return -primeT(first_der,dt)


def interpolate(c,v_max,dt,device):
    '''Given a curve c, max speed v_max and a discretization step dt, return interpolated curve'''
    d,n = c.shape
    sb = c[:, 0].to(device)
    r = 0
    d_max = v_max * dt
    for i in range(0,n - 1):
        crt_vect = c[:, i + 1]-c[:, i]
        crt_dist = torch.norm(crt_vect,2)
        u = crt_vect / crt_dist
        n_step = (int)(torch.floor((crt_dist - r) / d_max).item())
        if n_step > 0:
            interp_add = ((c[:,i] + r*u).view(-1,1).repeat(1,n_step+1) + (d_max*u).view(-1,1).repeat(1,n_step+1)*\
                         torch.Tensor(range(0,n_step+1)).repeat(d,1)).to(device)
            sb = torch.cat((sb.view(d,-1),interp_add),1)
            r = d_max - (torch.norm(sb[:, -1] - c[:, i+1],2))
        else:
            r = r - crt_dist
        print(sb.device)
    return sb



def isnan(x):
    return torch.sum(torch.isnan(x)).item()
