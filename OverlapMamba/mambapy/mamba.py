import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from pscan import pscan
import time
"""

This file closely follows the mamba_simple.py from the official Mamba implementation, and the mamba-minimal by @johnma2006.
The major differences are :
-the convolution is done with torch.nn.Conv1d
-the selective scan is done in PyTorch

A sequential version of the selective scan is also available for comparison.

- A Mamba model is composed of several layers, which are ResidualBlock.
- A ResidualBlock is composed of a MambaBlock, a normalization, and a residual connection : ResidualBlock(x) = mamba(norm(x)) + x
- This leaves us with the MambaBlock : its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).
First, we expand x into (B, L, 2*ED) (where E is usually 2) and split it into x and z, each (B, L, ED).
Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
We then multiply it by silu(z).
See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

"""
n = 0
sum = 0

@dataclass
class MambaConfig:
    d_model: int # D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    bias: bool = False
    conv_bias: bool = True

    pscan: bool = True # use parallel scan mode or sequential mode when training

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([
            ResidualBlock(config) for _ in range(config.n_layers)])
        #self.norm_f = RMSNorm(config.d_model)

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        for layer in self.layers:
            x = layer(x)

        #x = self.norm_f(x)

        return x
    
    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches

class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model)

    def forward(self, x):
        # x : (B, L, D)

        # output : (B, L, D)

        output = self.mixer(self.norm(x)) + x
        return output
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs: (B, ED, d_conv-1)

        # output : (B, D)
        # cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache



def shift(tensor, dim, index):
    length = tensor.size(dim)
    shifted_tensor = torch.cat((tensor.narrow(dim, index, length - index),
                                tensor.narrow(dim, 0, index)), dim=dim)
    return shifted_tensor

def unshift(tensor, dim, index):
    length = tensor.size(dim)
    unshifted_tensor = torch.cat((tensor.narrow(dim, length - index, index),
                                  tensor.narrow(dim, 0, length - index)), dim=dim)
    return unshifted_tensor



class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig, use_bimamba=True, use_shift=True):
        super().__init__()

        self.config = config
        self.use_bimamba = use_bimamba
        self.use_shift = use_shift

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                              kernel_size=config.d_conv, bias=config.conv_bias, 
                              groups=config.d_inner,
                              padding=config.d_conv - 1)
        self.conv1d_2 = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                  kernel_size=config.d_conv, bias=config.conv_bias,
                                  groups=config.d_inner,
                                  padding=config.d_conv - 1)
        # self.conv1d_3 = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
        #                           kernel_size=config.d_conv, bias=config.conv_bias,
        #                           groups=config.d_inner,
        #                           padding=config.d_conv - 1)
        # self.conv1d_4 = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
        #                           kernel_size=config.d_conv, bias=config.conv_bias,
        #                           groups=config.d_inner,
        #                           padding=config.d_conv - 1)
        
        # projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        self.x_proj_2 = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        # self.x_proj_3 = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        # self.x_proj_4 = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        self.dt_proj_2 = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        # self.dt_proj_3 = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        # self.dt_proj_4 = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
            nn.init.uniform_(self.dt_proj_2.weight, -dt_init_std, dt_init_std)
            # nn.init.uniform_(self.dt_proj_3.weight, -dt_init_std, dt_init_std)
            # nn.init.uniform_(self.dt_proj_4.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # dt bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            self.dt_proj_2.bias.copy_(inv_dt)
            # self.dt_proj_3.bias.copy_(inv_dt)
            # self.dt_proj_4.bias.copy_(inv_dt)
        #self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.D = nn.Parameter(torch.ones(config.d_inner))

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x):
        # x : (B, L, D)
        
        # y : (B, L, D)

        _, L, _ = x.shape

        xz = self.in_proj(x) # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1) # (B, L, ED), (B, L, ED)

        if not self.use_bimamba:
            # x branch
            x = x.transpose(1, 2) # (B, ED, L)
            x = self.conv1d(x)[:, :, :L] # depthwise convolution over time, with a short filter
            x = x.transpose(1, 2) # (B, L, ED)

            x = F.silu(x)
            y = self.ssm(x, 1)

            # z branch
            z = F.silu(z)

            output = y * z
            output = self.out_proj(output) # (B, L, D)

            return output

        x_f1 = x
        x_b1 = torch.flip(x, dims=[1])

        # x branch
        x_f = x_f1.transpose(1, 2)  # (B, ED, L)
        x_b = x_b1.transpose(1, 2)  #  (B, ED, L)


        x_f = self.conv1d(x_f)[:, :, :L] # depthwise convolution over time, with a short filter
        x_b = self.conv1d(x_b)[:, :, :L]


        x_f = x_f.transpose(1, 2)  # (B, L, ED)
        x_b = x_b.transpose(1, 2)  #  (B, L, ED)

        x_f = F.silu(x_f)
        x_b = F.silu(x_b)

        t = time.time()
        y_f = self.ssm(x_f, 1)
        a = time.time() - t
        b = a
        b /= 2
        b *= 4

        a /= 2
        a /= 30
        a *= 4

        b -= a
        b *= 1000
        #print(2.3 - b)
        if b >= 0.5:
            global n
            global sum
            tmp = 2.3 - b
            sum += tmp
            n += 1
            print(sum/n)

        y_b = self.ssm(x_b, 1)
        #print(f'cost:{time.time() - t:.8f}s')




        # z branch
        z = F.silu(z)

        y_f = y_f * z
        y_b = y_b * z
        y_b = torch.flip(y_b, dims=[1])

        if self.use_shift:
            index = int(torch.rand(1).item() * 900)

            x_fs = shift(x_f1, 1, index)
            x_bs = torch.flip(x_fs, dims=[1])

            x_fs = x_fs.transpose(1, 2)  #  (B, ED, L)
            x_bs = x_bs.transpose(1, 2)

            x_fs = self.conv1d_2(x_fs)[:, :, :L]  #  depthwise convolution over time, with a short filter
            x_bs = self.conv1d_2(x_bs)[:, :, :L]

            x_fs = x_fs.transpose(1, 2)  #  (B, L, ED)
            x_bs = x_bs.transpose(1, 2)  #  (B, L, ED)

            x_fs = F.silu(x_fs)
            x_bs = F.silu(x_bs)

            y_fs = self.ssm(x_fs, 2)
            y_bs = self.ssm(x_bs, 2)

            y_fs = y_fs * z
            y_bs = y_bs * z

            y_bs = torch.flip(y_bs, dims=[1])

            y_fs = unshift(y_fs, dim=1, index=index)
            y_bs = unshift(y_bs, dim=1, index=index)

            output = (y_f + y_b + y_fs + y_bs)


            output = self.out_proj(output)
        else:
            output = self.out_proj(y_f + y_b) # (B, L, D)

        return output
    
    def ssm(self, x, index):
        A = -torch.exp(self.A_log.float()) # (ED, N)
        D = self.D.float()
        # TODO remove .float()
        if index == 1:
            deltaBC = self.x_proj(x) # (B, L, dt_rank+2*N)
            delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
            delta = F.softplus(self.dt_proj(delta)) # (B, L, ED)
        elif index == 2:
            deltaBC = self.x_proj_2(x)  #  (B, L, dt_rank+2*N)
            delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                      dim=-1)  #  (B, L, dt_rank), (B, L, N), (B, L, N)
            delta = F.softplus(self.dt_proj_2(delta))  #  (B, L, ED)
        #
        # elif index == 3:
        #     deltaBC = self.x_proj_3(x)  #  (B, L, dt_rank+2*N)
        #     delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
        #                               dim=-1)  #  (B, L, dt_rank), (B, L, N), (B, L, N)
        #     delta = F.softplus(self.dt_proj_3(delta))  #  (B, L, ED)
        #
        # elif index == 4:
        #     deltaBC = self.x_proj_4(x)  #  (B, L, dt_rank+2*N)
        #     delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
        #                               dim=-1)  #  (B, L, dt_rank), (B, L, N), (B, L, N)
        #     delta = F.softplus(self.dt_proj_4(delta))  #  (B, L, ED)

        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)
        return y
    
    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)
        
        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    
    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
            
        hs = torch.stack(hs, dim=1) # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    
    # -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs : (B, ED, d_conv-1)
        
        # y : (B, D)
        # cache : (h, inputs)
        
        h, inputs = cache
        
        xz = self.in_proj(x) # (B, 2*ED)
        x, z = xz.chunk(2, dim=1) # (B, ED), (B, ED)

        # x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv-1] # (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2) # (B, ED, d_conv-1)
        cache = (h, inputs)
        
        return output, cache

    def ssm_step(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        A = -torch.exp(self.A_log.float()) # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()
        # TODO remove .float()

        deltaBC = self.x_proj(x) # (B, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, dt_rank), (B, N), (B, N)
        delta = F.softplus(self.dt_proj(delta)) # (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1) # (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)

        h = deltaA * h + BX # (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2) # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        # todo : pq h.squeeze(1) ??
        return y, h.squeeze(1)

# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
