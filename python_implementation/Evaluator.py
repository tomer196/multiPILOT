from abc import ABC, abstractmethod
import torch
from python_implementation import utils

class Evaluator(ABC):
    '''
    abstract evaluator class - suite of constraint distance metrics
    '''
    def __init__(self,eps = 1e-10):
        self.eps = eps
    @abstractmethod
    def norm(self,curve):
        '''measure curve to constraint distance'''
        return

    @abstractmethod
    def f_space(self,curve):
        '''measure curve to constraint distance over every temporal sample'''
        return

    @abstractmethod
    def dual(self,curve):
        '''dual phase cost calculation over set of curves (curve per constraint)'''
        return

    @abstractmethod
    def prox(self,q,alpha):
        '''calculate prox operator'''

class L1Norm(Evaluator):
    '''L1 norm'''
    def norm(self,curve):
        return torch.sum(torch.abs(curve))
    def f_space(self,curve):
        return torch.sum(torch.abs(curve),dim=1)
    def dual(self,curve):
        return torch.max(torch.abs(curve)).item()
    def prox(self,q,alpha):
        return q-utils.ProjectOntoL1Ball(q,torch.ones(q.shape),alpha)


class L2Norm(Evaluator):
    '''L2 norm'''
    def norm(self,curve):
        return torch.norm(curve,2)
    def f_space(self,curve):
        return torch.sqrt(torch.sum(curve**2, dim=1))
    def dual(self,curve):
        return self.norm(curve)
    def prox(self,q,alpha):
        return torch.nn.ReLU()(q*(1-alpha/torch.sqrt(torch.sum(q**2)+self.eps)))

class LInfNorm(Evaluator):
    '''inf norm'''
    def norm(self,curve):
        return torch.norm(curve,'inf')
    def f_space(self,curve):
        return torch.max(curve.view(curve.shape[0],-1),dim=1)
    def dual(self,curve):
        return torch.sum(torch.abs(curve))
    def prox(self,q,alpha):
        return torch.sign(q)*torch.max(torch.abs(q)-alpha,0)

class LInf2Norm(Evaluator):
    '''Combination of inf and L2 norm'''
    def norm(self,curve):
        return torch.max(torch.sqrt(torch.sum(curve**2,dim=2)),dim=1).values
    def f_space(self,curve):
        return torch.sqrt(torch.sum(curve**2,dim=2))
    def dual(self,curve):
        return torch.sum(self.f_space(curve),dim=1).view(-1,1)
    def prox(self,q,alpha):
        normed = torch.sqrt(torch.sum(q**2,dim=2)).view(q.shape[0],-1,1)
        rep = torch.cat((normed, normed), dim=2)
        #rep = torch.sqrt(torch.sum(q**2,dim=1)).repeat(1,q.shape[1])
        normed += self.eps
        rep_eps = torch.cat((normed,normed),dim=2)
        #rep_eps = torch.sqrt(torch.sum(q**2,dim=1)+self.eps).repeat(1,q.shape[1])
        return q/rep_eps.view(q.shape)*torch.nn.ReLU()(rep.view(q.shape)-alpha)
