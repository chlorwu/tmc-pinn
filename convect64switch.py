import time
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import argparse
import numpy as np
from collections import deque
from torch.optim import LBFGS
from tqdm import tqdm
from util import *
from model_dict import get_model

# =======================
# CONFIG
# =======================
TOTAL_EPOCHS = 50000

START_DTYPE = torch.float64
SWITCH_DTYPE = torch.float32

PLATEAU_WINDOW = 100
PLATEAU_EPS = 1e-4

step_size = 1e-4
num_step = 5
beta = 50
# =======================

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='PINN')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
device = args.device

# =======================
# DATA
# =======================
res, b_left, b_right, b_upper, b_lower = get_data([0, 2*np.pi], [0,1], 401, 401)
res_test, _, _, _, _ = get_data([0, 2*np.pi], [0,1], 101, 101)

if args.model in ['PINNsFormer','PINNMamba']:
    res = make_time_sequence(res, num_step=num_step, step=step_size)
    b_left = make_time_sequence(b_left, num_step=num_step, step=step_size)
    b_right = make_time_sequence(b_right, num_step=num_step, step=step_size)
    b_upper = make_time_sequence(b_upper, num_step=num_step, step=step_size)
    b_lower = make_time_sequence(b_lower, num_step=num_step, step=step_size)
    res_test = make_time_sequence(res_test, num_step=num_step, step=step_size)

res = torch.tensor(res, dtype=START_DTYPE, requires_grad=True).to(device)
b_left = torch.tensor(b_left, dtype=START_DTYPE, requires_grad=True).to(device)
b_right = torch.tensor(b_right, dtype=START_DTYPE, requires_grad=True).to(device)
b_upper = torch.tensor(b_upper, dtype=START_DTYPE, requires_grad=True).to(device)
b_lower = torch.tensor(b_lower, dtype=START_DTYPE, requires_grad=True).to(device)

x_res, t_res = res[...,0:1], res[...,1:2]
x_left, t_left = b_left[...,0:1], b_left[...,1:2]
x_right, t_right = b_right[...,0:1], b_right[...,1:2]
x_upper, t_upper = b_upper[...,0:1], b_upper[...,1:2]
x_lower, t_lower = b_lower[...,0:1], b_lower[...,1:2]

# =======================
# MODEL
# =======================
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()

if args.model == 'QRes':
    model = get_model(args).Model(2,256,1,4).to(START_DTYPE).to(device)
elif args.model in ['PINNsFormer','PINNsFormer_Enc_Only']:
    model = get_model(args).Model(2,32,1,1).to(START_DTYPE).to(device)
else:
    model = get_model(args).Model(2,512,1,4).to(START_DTYPE).to(device)

model.apply(init_weights)

def make_optimizer():
    return LBFGS(
        model.parameters(),
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-8,
        tolerance_change=1e-10
    )

optim = make_optimizer()

print(model)
print("Parameters:", sum(p.numel() for p in model.parameters()))

# =======================
# LOGGING
# =======================
os.makedirs('./results', exist_ok=True)

loss_log = f'./results/1dconvection_{args.model}_loss_log.txt'
with open(loss_log,'w') as f:
    f.write('
