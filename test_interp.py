import matplotlib.pylab as P
import torch
import numpy as np

def h_poly(t):
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

def interp(x, y, xs):
  m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
  m = torch.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
  I = P.searchsorted(x[1:], xs)
  dx = (x[I+1]-x[I])
  hh = h_poly((xs-x[I])/dx)
  return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx


# Example
if __name__ == "__main__":
    x_orig = np.load(f'spiral/{16}int_spiral_low.npy')[0,:3101,:]
    x_orig = torch.tensor(x_orig, requires_grad=False).float()

    t = torch.arange(0, 3101).float()
    t1 = t[::20]
    x_sub = x_orig[::20, :]

    x_interp = torch.zeros_like(x_orig)
    x_interp[:, 0] = interp(t1, x_sub[:, 0], t)
    x_interp[:, 1] = interp(t1, x_sub[:, 1], t)

    P.scatter(x_sub[:, 0], x_sub[:, 1], label='Samples', color='purple')
    P.plot(x_interp[:, 0], x_interp[:, 1], label='Interpolated curve')
    P.plot(x_orig[:, 0], x_orig[:, 1], '--', label='True Curve')
    P.legend()
    P.show()