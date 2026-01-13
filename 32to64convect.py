import time
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS, Adam
from tqdm import tqdm
import argparse
import numpy as np
from util import *
from model_dict import get_model

# =======================
# CONFIG
# =======================
TOTAL_EPOCHS = 50000
SWITCH_EPOCH = 3000          # Adam → LBFGS
START_DTYPE = torch.float32
SWITCH_DTYPE = torch.float64

STEP_SIZE = 1e-4
NUM_STEP = 5
BETA = 50
# =======================

# =======================
# SEED
# =======================
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# =======================
# ARGS
# =======================
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='pinn')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
device = args.device

# =======================
# DATA
# =======================
res, b_left, b_right, b_upper, b_lower = get_data(
    [0, 2*np.pi], [0,1], 401, 401
)
res_test, b_left_test, _, _, _ = get_data(
    [0, 2*np.pi], [0,1], 101, 101
)

if args.model in ['PINNsFormer', 'PINNMamba']:
    res = make_time_sequence(res, NUM_STEP, STEP_SIZE)
    b_left = make_time_sequence(b_left, NUM_STEP, STEP_SIZE)
    b_right = make_time_sequence(b_right, NUM_STEP, STEP_SIZE)
    b_upper = make_time_sequence(b_upper, NUM_STEP, STEP_SIZE)
    b_lower = make_time_sequence(b_lower, NUM_STEP, STEP_SIZE)
    res_test = make_time_sequence(res_test, NUM_STEP, STEP_SIZE)

res = torch.tensor(res, dtype=START_DTYPE, requires_grad=True).to(device)
b_left = torch.tensor(b_left, dtype=START_DTYPE, requires_grad=True).to(device)
b_right = torch.tensor(b_right, dtype=START_DTYPE, requires_grad=True).to(device)
b_upper = torch.tensor(b_upper, dtype=START_DTYPE, requires_grad=True).to(device)
b_lower = torch.tensor(b_lower, dtype=START_DTYPE, requires_grad=True).to(device)

x_res, t_res = res[...,0:1], res[...,1:2]
x_left, t_left = b_left[...,0:1], b_left[...,1:2]
x_upper, t_upper = b_upper[...,0:1], b_upper[...,1:2]
x_lower, t_lower = b_lower[...,0:1], b_lower[...,1:2]

# =======================
# MODEL
# =======================
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()

if args.model == 'KAN':
    model = get_model(args).Model(
        width=[2,5,5,1],
        grid=5, k=3,
        grid_eps=1.0,
        noise_scale_base=0.25,
        device=device
    )
elif args.model == 'QRes':
    model = get_model(args).Model(
        in_dim=2, hidden_dim=256,
        out_dim=1, num_layer=4
    )
else:
    model = get_model(args).Model(
        in_dim=2, hidden_dim=512,
        out_dim=1, num_layer=4
    )

model.apply(init_weights)
model = model.to(START_DTYPE).to(device)

# =======================
# OPTIMIZERS
# =======================
adam = Adam(model.parameters(), lr=1e-4)

def make_lbfgs():
    return LBFGS(
        model.parameters(),
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-8,
        tolerance_change=1e-10
    )

optimizer = adam

# =======================
# LOGGING
# =======================
os.makedirs('./results', exist_ok=True)

loss_hist = []
grad_hist = []
time_hist = []
precision_hist = []

# =======================
# TRAINING
# =======================
for epoch in tqdm(range(TOTAL_EPOCHS), ncols=100):

    timing_fwd = [0.0]
    timing_bwd = [0.0]
    grad_norm_box = [0.0]

    def closure():
        optimizer.zero_grad()

        t0 = time.time()

        pred_res = model(x_res, t_res)
        pred_left = model(x_left, t_left)
        pred_upper = model(x_upper, t_upper)
        pred_lower = model(x_lower, t_lower)

        u_x = torch.autograd.grad(
            pred_res, x_res,
            torch.ones_like(pred_res),
            retain_graph=True, create_graph=True
        )[0]

        u_t = torch.autograd.grad(
            pred_res, t_res,
            torch.ones_like(pred_res),
            retain_graph=True, create_graph=True
        )[0]

        loss_res = torch.mean((u_t + BETA * u_x)**2)
        loss_bc  = torch.mean((pred_upper - pred_lower)**2)
        loss_ic  = torch.mean((pred_left[:,0] - torch.sin(x_left[:,0]))**2)

        loss = loss_res + loss_bc + loss_ic

        timing_fwd[0] = time.time() - t0

        t1 = time.time()
        loss.backward()
        timing_bwd[0] = time.time() - t1

        grad_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_sq += p.grad.norm().item()**2
        grad_norm_box[0] = grad_sq**0.5

        return loss, loss_res, loss_bc, loss_ic

    if isinstance(optimizer, LBFGS):
        loss, lr, lb, li = optimizer.step(closure)
    else:
        loss, lr, lb, li = closure()
        optimizer.step()

    # =======================
    # SWITCH
    # =======================
    if epoch == SWITCH_EPOCH:
        print(f"\n🔁 SWITCH Adam FP32 → LBFGS FP64 at epoch {epoch}\n")

        model = model.to(SWITCH_DTYPE)
        x_res = x_res.to(SWITCH_DTYPE)
        t_res = t_res.to(SWITCH_DTYPE)
        x_left = x_left.to(SWITCH_DTYPE)
        t_left = t_left.to(SWITCH_DTYPE)
        x_upper = x_upper.to(SWITCH_DTYPE)
        t_upper = t_upper.to(SWITCH_DTYPE)
        x_lower = x_lower.to(SWITCH_DTYPE)
        t_lower = t_lower.to(SWITCH_DTYPE)

        optimizer = make_lbfgs()

    precision = 'fp32' if model.parameters().__next__().dtype == torch.float32 else 'fp64'

    loss_hist.append(loss.item())
    grad_hist.append(grad_norm_box[0])
    time_hist.append(timing_fwd[0] + timing_bwd[0])
    precision_hist.append(precision)

# =======================
# SAVE MODEL
# =======================
torch.save(model.state_dict(), './results/convect_adam32_lbfgs64.pt')

# =======================
# PLOTS (WITH SWITCH MARKER)
# =======================
epochs = np.arange(len(loss_hist))

plt.figure(figsize=(6,4))
plt.semilogy(epochs, loss_hist)
plt.axvline(SWITCH_EPOCH, color='red', linestyle='--', label='Precision / Optim Switch')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.legend()
plt.tight_layout()
plt.savefig('./results/loss_with_switch.pdf')
plt.close()

plt.figure(figsize=(6,4))
plt.plot(epochs, grad_hist)
plt.axvline(SWITCH_EPOCH, color='red', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Gradient Norm')
plt.tight_layout()
plt.savefig('./results/grad_norm_with_switch.pdf')
plt.close()

plt.figure(figsize=(6,4))
plt.plot(epochs, time_hist)
plt.axvline(SWITCH_EPOCH, color='red', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Time per Epoch (s)')
plt.tight_layout()
plt.savefig('./results/time_with_switch.pdf')
plt.close()
