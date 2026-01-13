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
SWITCH_EPOCH = 3000          # Adam FP32 → LBFGS FP64
START_DTYPE = torch.float32
SWITCH_DTYPE = torch.float64

STEP_SIZE = 1e-4
NUM_STEP = 5
BETA = 50
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

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
res, b_left, b_right, b_upper, b_lower = get_data([0, 2*np.pi], [0,1], 401, 401)
res_test, b_left_test, _, _, _ = get_data([0, 2*np.pi], [0,1], 101, 101)

if args.model in ['PINNsFormer','PINNMamba']:
    res = make_time_sequence(res, NUM_STEP, STEP_SIZE)
    b_left = make_time_sequence(b_left, NUM_STEP, STEP_SIZE)
    b_right = make_time_sequence(b_right, NUM_STEP, STEP_SIZE)
    b_upper = make_time_sequence(b_upper, NUM_STEP, STEP_SIZE)
    b_lower = make_time_sequence(b_lower, NUM_STEP, STEP_SIZE)
    res_test = make_time_sequence(res_test, NUM_STEP, STEP_SIZE)

def to_tensor(x, dtype):
    return torch.tensor(x, dtype=dtype, requires_grad=True).to(device)

res = to_tensor(res, START_DTYPE)
b_left = to_tensor(b_left, START_DTYPE)
b_right = to_tensor(b_right, START_DTYPE)
b_upper = to_tensor(b_upper, START_DTYPE)
b_lower = to_tensor(b_lower, START_DTYPE)

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
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()

if args.model == 'KAN':
    model = get_model(args).Model(
        width=[2,5,5,1], grid=5, k=3,
        grid_eps=1.0, noise_scale_base=0.25,
        device=device
    ).to(START_DTYPE)
elif args.model == 'QRes':
    model = get_model(args).Model(2, 256, 1, 4)
else:
    model = get_model(args).Model(2, 512, 1, 4)

model.apply(init_weights)
model = model.to(device).to(START_DTYPE)

# =======================
# OPTIMIZERS
# =======================
adam = Adam(model.parameters(), lr=1e-4)

def make_lbfgs():
    return LBFGS(
        model.parameters(),
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-8,
        tolerance_change=1e-10
    )

optimizer = adam

# =======================
# FLOPs (same logic as reaction)
# =======================
def estimate_flops(model, batch):
    flops = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            flops += 2 * m.in_features * m.out_features * batch
    return flops

batch_size = x_res.shape[0]
base_flops = estimate_flops(model, batch_size)

# =======================
# LOG BUFFERS
# =======================
epochs = []
loss_res_log = []
loss_bc_log = []
loss_ic_log = []
loss_total_log = []
grad_norm_log = []
time_log = []
flops_sec_log = []
precision_log = []

# =======================
# TRAINING LOOP
# =======================
for epoch in tqdm(range(TOTAL_EPOCHS), ncols=100):

    timing_fwd, timing_bwd = 0.0, 0.0
    grad_norm = 0.0

    def closure():
        nonlocal timing_fwd, timing_bwd, grad_norm
        optimizer.zero_grad()

        t0 = time.time()

        pred_res = model(x_res, t_res)
        pred_left = model(x_left, t_left)
        pred_upper = model(x_upper, t_upper)
        pred_lower = model(x_lower, t_lower)

        u_x = torch.autograd.grad(pred_res, x_res,
                                  torch.ones_like(pred_res),
                                  retain_graph=True, create_graph=True)[0]
        u_t = torch.autograd.grad(pred_res, t_res,
                                  torch.ones_like(pred_res),
                                  retain_graph=True, create_graph=True)[0]

        loss_res = torch.mean((u_t + BETA*u_x)**2)
        loss_bc = torch.mean((pred_upper - pred_lower)**2)
        loss_ic = torch.mean((pred_left[:,0] - torch.sin(x_left[:,0]))**2)

        loss = loss_res + loss_bc + loss_ic

        timing_fwd = time.time() - t0
        t1 = time.time()
        loss.backward()
        timing_bwd = time.time() - t1

        grad_norm = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None))
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

    precision = "fp32" if model.parameters().__next__().dtype == torch.float32 else "fp64"
    total_time = timing_fwd + timing_bwd
    flops = base_flops * (2 if precision == "fp64" else 1)
    flops_sec = flops / total_time if total_time > 0 else 0.0

    # =======================
    # LOG
    # =======================
    epochs.append(epoch)
    loss_res_log.append(lr.item())
    loss_bc_log.append(lb.item())
    loss_ic_log.append(li.item())
    loss_total_log.append(loss.item())
    grad_norm_log.append(grad_norm.item())
    time_log.append(total_time)
    flops_sec_log.append(flops_sec)
    precision_log.append(precision)

# =======================
# SAVE MODEL
# =======================
torch.save(model.state_dict(), f"{RESULTS_DIR}/1dconvection_adam32_lbfgs64.pt")

# =======================
# PLOTS
# =======================
def save_plot(y, title, ylabel, fname, logy=False):
    plt.figure()
    plt.plot(epochs, y)
    if logy:
        plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"{RESULTS_DIR}/{fname}", dpi=200)
    plt.close()

save_plot(loss_total_log, "Total Loss", "Loss", "loss_total.png", logy=True)
save_plot(loss_res_log, "Residual Loss", "Loss", "loss_res.png", logy=True)
save_plot(loss_bc_log, "BC Loss", "Loss", "loss_bc.png", logy=True)
save_plot(loss_ic_log, "IC Loss", "Loss", "loss_ic.png", logy=True)
save_plot(grad_norm_log, "Gradient Norm", "||∇L||", "grad_norm.png", logy=True)
save_plot(time_log, "Time per Epoch", "Seconds", "epoch_time.png")
save_plot(flops_sec_log, "FLOPs/sec", "FLOPs/s", "flops_sec.png")
