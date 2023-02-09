import torch
import numpy
import scipy,scipy.linalg

from torch_geometric.utils import to_dense_adj

train = False

def softmin(b): return -0.5*torch.log(1.0+torch.exp(-2*b))

class Block(torch.nn.Module):
    def __init__(self, s, k):
        super().__init__()
        self.W = []
        self.B = []
        for a,b in zip(s[:-1],s[1:]):
            self.W += [
                torch.nn.Parameter(torch.FloatTensor([
                    numpy.random.normal(0,a**-.5,[a,b]) for _ in range(k)
                ]))
            ]
            self.B += [
                torch.nn.Parameter(torch.FloatTensor([
                    numpy.random.normal(0,1,[b]) for _ in range(k)
                ]))
            ]
        self.W = torch.nn.ParameterList(self.W)
        self.B = torch.nn.ParameterList(self.B)

    def forward(self,Hin,A,mask=None):

        n = Hin.shape[0]
        device = Hin.device
        if mask is None:
            mask = torch.ones((1,), device=device)

        # FIXME: doesn't work for len(self.W) == 1
        for Wo,Bo,Ao in zip(self.W,self.B,[A*mask]+[torch.eye(n, device=device).reshape(1,n,n)]*(len(self.W)-1)):

            Hout = torch.zeros((1,), device=device)
            for ao,wo,bo in zip(Ao,Wo,Bo):
                bo = softmin(bo)
                Hout = Hout + ao.permute(1,0).matmul(Hin).matmul(wo) + bo

            if Wo.shape[2] > 10:
                Hin = Hout.clamp(min=0)
            else:
                Hin = Hout

        return Hin

    def lrpforward(self,Hin,A,gamma):
        n = Hin.shape[0]
        device = Hin.device
        for Wo,Bo,Ao in zip(self.W,self.B,[A]+[torch.eye(n).reshape(1,n,n)]*(len(self.W)-1)):
            Hout = torch.zeros((1,), device=device)
            Pout = torch.FloatTensor(1e-6, device=device)
            for ao,wo,bo in zip(Ao,Wo,Bo):
                bo = softmin(bo)
                Hout = Hout + ao.permute(1,0).matmul(Hin).matmul(wo) + bo

                if gamma > 0 and wo.shape[-1] > 10:  
                    wp = wo + gamma*wo.clamp(min=0)
                    bp = bo + gamma*bo.clamp(min=0)
                    Pout = Pout + ao.permute(1,0).matmul(Hin).matmul(wp) + bp

            if gamma > 0 and wo.shape[-1] > 10:  
                Hout = Pout * (Hout / Pout).data

            if Wo.shape[2] > 10:
                Hin = Hout.clamp(min=0)
            else:
                Hin = Hout

        return Hin

class GNN(torch.nn.Module):

    def __init__(self,*sizes,mode='std'):
        super().__init__()
        if mode == 'std': k=1
        if mode == 'cheb': k=3

        self.blocks = torch.nn.ModuleList([Block([s, s, s],k) for s in sizes])
        self.mode = mode

    def __call__(self, x, edge_index, masks=None):
        """
        A = to_dense_adj(edge_index)
        H0 = x.unsqueeze(0)
        return self.forward(A, H0=H0, masks=masks)
        """
        device = edge_index.device
        A = to_dense_adj(edge_index)[0].to(device)
        return self.forward(A, H0=x, masks=masks)

    def adj(self,A):
        if self.mode == 'std':
            L1 = A / 2
            A = torch.cat((L1.unsqueeze(0),)) 
            return A/1**.5

        if self.mode == 'cheb':
            L0 = torch.eye(len(A), device=A.device)
            L1 = A / 2
            L2 = L1.matmul(L1)
            A = torch.cat((L0.unsqueeze(0),L1.unsqueeze(0),L2.unsqueeze(0)))
            return A/3**.5

    def ini(self,A,H0):
        if H0 == None: H0 = torch.ones([len(A),1])
        return H0

    def forward(self, A, H0=None, masks=None):
        if masks is None:
            masks = [1]*(len(self.blocks)-1)
        H0 = self.ini(A, H0)

        H = self.blocks[0].forward(H0,torch.eye(H0.shape[0], device=A.device).unsqueeze(0))

        A = self.adj(A)

        for l,mask in zip(self.blocks[1:], masks):
            H = l.forward(H, A, mask=mask)

        # H = H.sum(dim=0) / 20**.5
        return H

    def lrp(self, A, gammas, t, inds, H0=None):
        
        H0 = self.ini(A, H0)
        H1 = self.blocks[0].forward(H0,torch.eye(H0.shape[0], device=A.device).unsqueeze(0)).data

        A = self.adj(A)

        H1.requires_grad_(True)

        H = H1

        if inds is None:
            for l,gamma in zip(self.blocks[1:],gammas):
                H = l.lrpforward(H,A,gamma)
        else:
            
            for l,i,gamma in zip(self.blocks[1:],inds,gammas):
                H = l.lrpforward(H,A,gamma)
                M = torch.FloatTensor(numpy.eye(H.shape[0])[i][:,numpy.newaxis])
                H = H * M + (1-M) * (H.data)

        H = H.sum(dim=0) / 20**.5

        H[t].backward()
        return (H1*H1.grad).sum(dim=1).data
