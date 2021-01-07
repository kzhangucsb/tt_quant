

import torch
import torch.nn as nn
from torch.nn import Parameter, ParameterList
import tt_nn as tt_nn
import numpy as np

class tt_nn_fcn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias, shift, *weights):
        dim = len(weights)
        # print('forward')
        ret, maxv = tt_nn.forward(input, weights, bias, shift)
        ctx.save_for_backward(maxv, input, shift, *weights)
        # print(maxv)
        return ret

    @staticmethod
    def backward(ctx, grad_out):
        maxv = ctx.saved_tensors[0]
        input = ctx.saved_tensors[1]
        shift = ctx.saved_tensors[2]
        weights = ctx.saved_tensors[3:]
        dim = len(weights)
        ret = tt_nn.backward(grad_out, input, weights, shift)
        ret[2] += maxv
        #print(maxv)
        return tuple(ret)

   
class TTlinear(nn.Module):
    def __init__(self, in_size, out_size, rank, 
                 prior_type=None, prior_para=[], em_stepsize=0.1, **kwargs):
        # increase beta to decrease rank
        super(TTlinear, self).__init__()
        assert(len(in_size) == len(out_size))
        assert(len(rank) == len(in_size) - 1)
        self.in_size = list(in_size)
        self.out_size = list(out_size)
        self.rank = list(rank)
        self.factors = ParameterList()
        r1 =[1] + list(rank)
        r2 = list(rank) + [1]
        for ri, ro, si, so in zip(r1, r2, in_size, out_size):
            p = Parameter(torch.Tensor(ri, so, si, ro))
            self.factors.append(p)
        self.bias = Parameter(torch.Tensor(np.prod(out_size)))
        self._initialize_weights()
        
        self.prior_type = prior_type
        self.prior_para = prior_para.copy()
        self.rank_parameters = ParameterList([
            Parameter(torch.ones(r), requires_grad=False) for r in rank])
        self.em_stepsize = em_stepsize
        self.mask = ParameterList([
            Parameter(torch.ones(r), requires_grad=False) for r in rank])
        
        dim = len(in_size)
        self.register_buffer('shift', torch.zeros(dim * 2 + dim * (dim + 1) // 2, 
                                 requires_grad=True))
        self.shift_adj = torch.zeros_like(self.shift)
        

    def forward(self, x):
        return tt_nn_fcn.apply(x, self.bias, self.shift, *list(self.factors))

    def _initialize_weights(self):
        for f in self.factors:
            # scale = np.sqrt(3.0 / f.shape[3] / f.shape[2])
            scale = 0.7
            nn.init.uniform_(f, -scale, scale)
        # self.factors[-1].data.mul_(np.sqrt(2.0))
        nn.init.constant_(self.bias, 0)
        
    def adj_shift(self, ths):
        if self.shift.grad is None:
            return
        for i in range(len(self.factors) * 2 + 1):
            if self.shift.grad[i] > 0.9:
                self.shift_adj[i] = max(self.shift_adj[i], 0)
                self.shift_adj[i] += 1
            elif self.shift.grad[i] < 0.3:
                self.shift_adj[i] = min(self.shift_adj[i], 0)
                self.shift_adj[i] -= 1
            else:
                self.shift_adj[i] = 0
        for i in range(len(self.factors) * 2 + 1, len(self.shift)):
            if self.shift.grad[i] > 0.9 * 128:
                self.shift_adj[i] = max(self.shift_adj[i], 0)
                self.shift_adj[i] += 1
            elif self.shift.grad[i] < 0.3 * 128:
                self.shift_adj[i] = min(self.shift_adj[i], 0)
                self.shift_adj[i] -= 1
            else:
                self.shift_adj[i] = 0
        
        for i in range(len(self.factors)):
            if self.shift_adj[i] > 1:
                self.shift.data[i] += 1
                # print(f'pos {i} increase')
            if self.shift_adj[i] < -1:
                self.shift.data[i] -= 1
                # print(f'pos {i} decrease')
         
        for i in range(len(self.shift)):
            if self.shift_adj[i] > ths:
                self.shift.data[i] += 1
                # print(f'pos {i} increase')
            if self.shift_adj[i] < -ths:
                self.shift.data[i] -= 1
                # print(f'pos {i} decrease')
        
                
        self.shift.grad.zero_()
        pass
        
    def get_rank_parameters_update(self):
        updates = []
        realrank = self.report_rank()
        realrank = [1] + realrank + [1]

        for i in range(len(self.rank)):

            M = torch.sum(self.factors[i]**2, dim=[0,1,2])
            D = self.in_size[i] * self.out_size[i] * realrank[i]
            N = self.in_size[i] * self.out_size[i] * realrank[i] \
                    + self.in_size[i+1] * self.out_size[i+1] * realrank[i+2]
            if self.prior_type == 'log_uniform':
                update = M / (D + 1)
            elif self.prior_type == 'gamma':
                update = (2 * self.prior_para[1] + M) / (D + 2 + 2 * self.prior_para[0])

            elif self.prior_type == 'half_cauchy':
                update = (M - (self.prior_para[0]**2) * D +
                          torch.sqrt(M**2 + (M * self.prior_para[0]**2) * (2.0 * D + 8.0) +
                                  (D**2.0) * (self.prior_para[0]**4.0))) / (2 * D + 4.0)
                
            elif self.prior_type == 'l2p' or self.prior_type == 'l21_prox' \
                or self.prior_type == 'l21_prox_gamma':

                update = (M , self.out_size[i], N)

            else:
                assert False, 'Unknown prior type'
            updates.append(update)

        return updates

        
    def update_rank_parameters(self, stepsize=None):
        if self.prior_type is None:
            return 
        
        self.apply_mask()
        with torch.no_grad():
            rank_updates = self.get_rank_parameters_update()  
            if self.prior_type == 'l21_prox_gamma' and len(self.prior_para) == 2:
                self.prior_para.append(0)              
            for rank_parameter, update, factor in zip(self.rank_parameters, rank_updates, self.factors[:-1]):
                
                if self.prior_type == 'l21_prox':
                    M, D, N = update
                    scale = torch.clamp_min(1 - stepsize * self.prior_para[0] * N/
                                            (torch.sqrt(M * D) ), 0)
                    rank_parameter.data.copy_(M / D * scale**2)
                    factor.data.mul_(scale.view(1, 1, 1, -1))
                elif self.prior_type == 'l21_prox_gamma':
                    M, D, N = update
                    scale = torch.clamp_min(1 - stepsize * self.prior_para[2] * N/
                                            (torch.sqrt(M * D) ), 0)
                    rank_parameter.data.copy_(M / D * scale**2)

                    factor.data.mul_(scale.view(1, 1, 1, -1))
                elif self.prior_type == 'l2p':
                    rank_parameter.data.copy_(update)
                else:
                    rank_parameter.data.mul_(1 - self.em_stepsize)
                    rank_parameter.data.add_(self.em_stepsize * update)
                    
            if self.prior_type == 'l21_prox_gamma':
                T = 0
                theta = self.prior_para[0]
                k = self.prior_para[1]
                for (M, D, N) in rank_updates:
                    T += torch.sum(N * torch.sqrt(M/D)).item()
                    
                self.prior_para[2] = (k-1) / (1/theta + T)
            
    def get_bayes_loss(self):
        if self.prior_type is None:
            return 0
        
        loss = 0
        if self.prior_type == 'l21_prox' or self.prior_type == 'l21_prox_gamma':
            return 0
        elif self.prior_type == 'l2p':
            updates = self.get_rank_parameters_update()
            p = self.prior_para[1] if len(self.prior_para) >= 2 else 1.0
            for update in updates:
                loss += torch.sum(torch.sqrt(update)**p)**(1/p) * self.prior_para[0]
            
        else: 
            for [factor, rank_parameter] in zip(self.factors[:-1], self.rank_parameters):
                loss += torch.sum(torch.sum(factor**2, dim=[0, 1, 2]) / rank_parameter)
                
        return loss
    
    def update_rank_mask(self, ths=1e-3):
        for (mask, rank_parameter) in zip(self.mask, self.rank_parameters):
            mask.copy_(torch.where(rank_parameter > ths, 
                                torch.ones_like(rank_parameter), 
                                torch.zeros_like(rank_parameter)))
    def apply_mask(self):
        with torch.no_grad():
            for i in range(len(self.rank)):
                self.factors[i].data.mul_(self.mask[i].view(1, 1, 1, -1))
                self.factors[i+1].data.mul_(self.mask[i].view(-1, 1, 1, 1))
    def report_rank(self):
        return [torch.sum(m).item() for m in self.mask]
    
        
        
    
