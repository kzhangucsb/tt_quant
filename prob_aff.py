#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 11:39:26 2020

@author: zkq
"""

import torch
import torch.nn as nn

def prob_affected_factors(model: nn.Module, inshape):
    para_dict = dict(model.named_parameters()) 
    
    # find shift
    shifts = dict()
    shifts_affected = dict()
    for n, p in model.state_dict().items():
        if 'shift' in n:
            shifts[n] = p
            p.data.zero_()
            shifts_affected[n] = [[] for _ in p]
            
    # base output
    model.zero_grad()
    inp = torch.rand([1, *inshape])
    outp = model(inp)
    out_base = torch.norm(outp)
    grad_base = dict()
    out_grad = torch.rand_like(outp)
    outp.backward(out_grad)
    for n, p in model.named_parameters():
        if p.grad is not None:
            grad_base[n] = torch.norm(p.grad)
            
    # find affected 
    for ns, s in shifts.items():
        for i in range(len(s)):
            model.zero_grad()
            for s1 in shifts.values():
                s1.data.zero_()
            s[i] = 1
            outp = model(inp)
            outp.backward(out_grad)
            if torch.norm(outp) == out_base:
                # print('output')
                pass
            elif torch.norm(outp) == out_base / 2:
                shifts_affected[ns][i].append('output')
            else:
                print(ns, i, 'output', out_base, torch.norm(outp))
                
            for n, pg in grad_base.items():
                if torch.norm(para_dict[n].grad) == pg:
                    # print(n)
                    pass
                elif torch.norm(para_dict[n].grad) == pg / 2:
                    shifts_affected[ns][i].append(n)
                else:
                    print(ns, i, n, pg, torch.norm(para_dict[n].grad))
                    
    return shifts_affected

def adj_output(model, aff, base_shift = 0):
    ret = 1
    state_dict = model.state_dict()
    for ns in aff:
        for s, aff_s in zip(state_dict[ns], aff[ns]):
            # print(aff_s)
            if 'output' in aff_s:
                ret = ret * (2 ** s)
                
    ret = ret / (2 ** base_shift)
    return ret

def adj_grad(model, aff):
    state_dict = model.state_dict()
    para_dict = dict(model.named_parameters()) 
    for ns in aff:
        for s, aff_s in zip(state_dict[ns], aff[ns]):
            for np in aff_s:
                if np == 'output':
                    continue
                para_dict[np].grad.mul_(2 ** s)
                
    
    

                
                
            
                
            
            
            
            
    
    
    
    
    
    
    