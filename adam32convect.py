import time
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import Adam
from tqdm import tqdm
import argparse
import numpy as np
from util import *
from model_dict import get_model

# =======================
# CONFIG
# =======================
TOTAL_EPOCHS = 50000
STEP_SIZE = 1e-4
NUM_STEP = 5
BETA = 50
DTYPE = torch.float32  # Adam32 only

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

res = torch.tensor(res, dtype=DTYPE, requires_grad=True).to(device)
b_left = torch.tensor(b_left, dtype=DTYPE, requires_grad=True).to(device)
b_right = torch.tensor(b_right, dtype=DTYPE, requires_grad=True).to(device)
b_upper = torch.tensor(b_upper, dtype=DTYPE, requires_grad=True).to(device)
b_lower = torch.tensor(b_lower, dtype=DTYPE, requires_grad=True).to(device)

x_res, t_res = res[...,0:1], res[...,1:2]
x_left, t_left = b_left[...,0:1], b_left[...,1:2]
x_right, t_right = b_right[...,0:1], b_right[...,1:2]
x_upper, t_upper = b_upper[...,0:1], b_upper[...,1:2]
x_lower, t_lower = b_lower[...,0:1], b_lower[...,1:2]

x_test, t_test = res_test[...,0:1], res_test[...,1:2]

# =======================
# MODEL
# =======================
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()

if args.model == 'KAN':
    model = get_model(args).Model(width=[2,5,5,1], grid=5, k=3,
                                  grid_eps=1.0, noise_scale_base=0.25,
                                  device=device).to(DTYPE)
elif args.model == 'QRes':
    model = get_model(args).Model(in_dim=2, hidden_dim=256,
                                  out_dim=1, num_layer=4)
else:
    model = get_model(args).Model(in_dim=2, hidden_dim=512,
                                  out_dim=1, num_layer=4)

model.apply(init_weights)
model = model.to(DTYPE).to(device)

# =======================
# OPTIMIZER
# =======================
optimizer = Adam(model.parameters(), lr=1e-4)

# =======================
# LOGGING SETUP
# =======================
os.makedirs('./results', exist_ok=True)

loss_log = './results/adam32_loss_log.txt'
grad_log = './results/adam32_grad_log.txt'
flops_log = './results/adam32_flops_log.txt'

with open(loss_log,'w') as f:
    f.write('epoch,loss_res,loss_bc,loss_ic,total_loss,precision\n')
with open(grad_log,'w') as f:
    f.write('epoch,grad_norm,grad_mean,grad_std,precision\n')
with open(flops_log,'w') as f:
    f.write('epoch,fwd_flops,bwd_flops,total_flops,fwd_time,bwd_time,total_time,flops_per_sec,precision\n')

# =======================
# FLOPs ESTIMATION
# =======================
def estimate_flops(model, batch):
    flops = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            flops += 2*m.in_features*m.out_features*batch
    return flops

batch_size = x_res.shape[0]
fwd_flops = estimate_flops(model, batch_size)
bwd_flops = 2*fwd_flops
total_flops = fwd_flops + bwd_flops

# =======================
# TRAINING LOOP
# =======================
loss_track = []
grad_stats = []

for epoch in tqdm(range(TOTAL_EPOCHS), ncols=100):
    timing = [0.0, 0.0]

    optimizer.zero_grad()
    t0 = time.time()

    # Forward
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

    timing[0] = time.time() - t0

    # Backward
    t1 = time.time()
    loss.backward()
    optimizer.step()
    timing[1] = time.time() - t1

    # Grad stats
    grad_norms, grad_means, grad_stds = [], [], []
    for p in model.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())
            grad_means.append(p.grad.mean().item())
            grad_stds.append(p.grad.std().item())

    grad_norm = np.sqrt(np.sum(np.array(grad_norms)**2))
    grad_mean = np.mean(np.array(grad_means))
    grad_std = np.mean(np.array(grad_stds))
    grad_stats.append({'norm': grad_norm, 'mean': grad_mean, 'std': grad_std})

    # Log in real-time
    precision = 'fp32'
    total_time = sum(timing)
    flops_sec = total_flops / total_time if total_time>0 else 0.0

    with open(loss_log,'a') as f:
        f.write(f"{epoch},{loss_res.item():.8e},{loss_bc.item():.8e},{loss_ic.item():.8e},{loss.item():.8e},{precision}\n")
        f.flush(); os.fsync(f.fileno())

    with open(grad_log,'a') as f:
        f.write(f"{epoch},{grad_norm:.8e},{grad_mean:.8e},{grad_std:.8e},{precision}\n")
        f.flush(); os.fsync(f.fileno())

    with open(flops_log,'a') as f:
        f.write(f"{epoch},{fwd_flops:.2e},{bwd_flops:.2e},{total_flops:.2e},{timing[0]:.6f},{timing[1]:.6f},{total_time:.6f},{flops_sec:.2e},{precision}\n")
        f.flush(); os.fsync(f.fileno())

    loss_track.append([loss_res.item(), loss_bc.item(), loss_ic.item(), loss.item()])

# =======================
# SAVE MODEL
# =======================
torch.save(model.state_dict(), './results/1dconvection_adam32.pt')

# =======================
# PREDICTION & GRAPHS
# =======================
with torch.no_grad():
    pred = model(x_test, t_test)[:,0:1].cpu().numpy().reshape(101,101)

def u_exact(x,t):
    return np.sin(x - BETA*t)

res_test_data, _, _, _, _ = get_data([0,2*np.pi],[0,1],101,101)
u = u_exact(res_test_data[:,0], res_test_data[:,1]).reshape(101,101)

# Errors
rl1 = np.sum(np.abs(u - pred)) / np.sum(np.abs(u))
rl2 = np.sqrt(np.sum((u - pred)**2) / np.sum(u**2))
print(f"relative L1 error: {rl1:.6f}, relative L2 error: {rl2:.6f}")

# Predicted u(x,t)
plt.figure(figsize=(4,3))
plt.imshow(pred, extent=[0,2*np.pi,1,0], aspect='auto')
plt.xlabel('x'); plt.ylabel('t'); plt.title('Predicted u(x,t)'); plt.colorbar(); plt.tight_layout()
plt.savefig('./results/adam32_pred.pdf'); plt.close()

# Exact
plt.figure(figsize=(4,3))
plt.imshow(u, extent=[0,2*np.pi,1,0], aspect='auto')
plt.xlabel('x'); plt.ylabel('t'); plt.title('Exact u(x,t)'); plt.colorbar(); plt.tight_layout()
plt.savefig('./results/adam32_exact.pdf'); plt.close()

# Absolute Error
plt.figure(figsize=(4,3))
plt.imshow(pred-u, extent=[0,2*np.pi,1,0], aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
plt.xlabel('x'); plt.ylabel('t'); plt.title('Absolute Error'); plt.colorbar(); plt.tight_layout()
plt.savefig('./results/adam32_error.pdf'); plt.close()

# Gradient plots
grad_norms_history = [s['norm'] for s in grad_stats]
grad_means_history = [s['mean'] for s in grad_stats]
grad_stds_history = [s['std'] for s in grad_stats]

plt.figure(figsize=(10,6))
plt.plot(grad_norms_history); plt.xlabel('Epoch'); plt.ylabel('Gradient Norm'); plt.title('Gradient Norms'); plt.grid(); plt.savefig('./results/adam32_grad_norms.pdf'); plt.close()
plt.figure(figsize=(10,6))
plt.plot(grad_means_history); plt.xlabel('Epoch'); plt.ylabel('Gradient Mean'); plt.title('Gradient Means'); plt.grid(); plt.savefig('./results/adam32_grad_means.pdf'); plt.close()
plt.figure(figsize=(10,6))
plt.plot(grad_stds_history); plt.xlabel('Epoch'); plt.ylabel('Gradient Std'); plt.title('Gradient Stds'); plt.grid(); plt.savefig('./results/adam32_grad_stds.pdf'); plt.close()
