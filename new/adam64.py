import os
import time
import random
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
from util import *
from model_dict import get_model

# =======================
# ARGPARSE
# =======================
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='PINN')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--epochs', type=int, default=50000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--beta', type=float, default=50)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
device = args.device
EPOCHS = args.epochs
LR = args.lr
BETA = args.beta
SEED = args.seed

# =======================
# SEED
# =======================
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# =======================
# RESULTS DIR
# =======================
os.makedirs('./results', exist_ok=True)
loss_log_file = f'./results/1dconvection_{args.model}_loss_log.csv'
flops_log_file = f'./results/1dconvection_{args.model}_flops_log.csv'
grad_log_file = f'./results/1dconvection_{args.model}_grad_log.csv'

for f in [loss_log_file, flops_log_file, grad_log_file]:
    with open(f, 'w') as ff:
        pass

# =======================
# DATA
# =======================
NUM_STEP = 5
STEP_SIZE = 1e-4

res, b_left, b_right, b_upper, b_lower = get_data([0, 2*np.pi], [0,1], 401, 401)
res_test, b_left_test, _, _, _ = get_data([0, 2*np.pi], [0,1], 101, 101)

if args.model in ['PINNsFormer', 'PINNMamba']:
    res = make_time_sequence(res, NUM_STEP, STEP_SIZE)
    b_left = make_time_sequence(b_left, NUM_STEP, STEP_SIZE)
    b_right = make_time_sequence(b_right, NUM_STEP, STEP_SIZE)
    b_upper = make_time_sequence(b_upper, NUM_STEP, STEP_SIZE)
    b_lower = make_time_sequence(b_lower, NUM_STEP, STEP_SIZE)
    res_test = make_time_sequence(res_test, NUM_STEP, STEP_SIZE)

# FP64 TENSORS
res = torch.tensor(res, dtype=torch.float64, requires_grad=True).to(device)
b_left = torch.tensor(b_left, dtype=torch.float64, requires_grad=True).to(device)
b_right = torch.tensor(b_right, dtype=torch.float64, requires_grad=True).to(device)
b_upper = torch.tensor(b_upper, dtype=torch.float64, requires_grad=True).to(device)
b_lower = torch.tensor(b_lower, dtype=torch.float64, requires_grad=True).to(device)

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
        nn.init.zeros_(m.bias)

if args.model == 'PINNsFormer' or args.model == 'PINNsFormer_Enc_Only':
    model = get_model(args).Model(in_dim=2, hidden_dim=32, out_dim=1, num_layer=1)
else:
    model = get_model(args).Model(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4)

model.apply(init_weights)
model = model.to(torch.float64).to(device)

# =======================
# OPTIMIZER
# =======================
optimizer = Adam(model.parameters(), lr=LR)

# =======================
# TRAINING LOOP
# =======================
loss_track = []
gradient_stats = []
flops_track = []

def estimate_flops(model, batch_size):
    flops = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            flops += (2*module.in_features*module.out_features + module.out_features) * batch_size
    return flops

fwd_flops = estimate_flops(model, x_res.shape[0])
bwd_flops = 2 * fwd_flops
total_flops = fwd_flops + bwd_flops

for epoch in tqdm(range(EPOCHS), ncols=100):
    optimizer.zero_grad()
    t0 = time.time()
    pred_res = model(x_res, t_res)
    pred_left = model(x_left, t_left)
    pred_right = model(x_right, t_right)
    pred_upper = model(x_upper, t_upper)
    pred_lower = model(x_lower, t_lower)

    u_x = torch.autograd.grad(pred_res, x_res, torch.ones_like(pred_res), create_graph=True)[0]
    u_t = torch.autograd.grad(pred_res, t_res, torch.ones_like(pred_res), create_graph=True)[0]

    loss_res = torch.mean((u_t + BETA * u_x)**2)
    loss_bc = torch.mean((pred_upper - pred_lower)**2)
    loss_ic = torch.mean((pred_left[:,0] - torch.sin(x_left[:,0]))**2)

    loss = loss_res + loss_bc + loss_ic
    fwd_time = time.time() - t0

    t1 = time.time()
    loss.backward()
    bwd_time = time.time() - t1
    optimizer.step()

    # Gradient stats
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().view(-1))
    grads = torch.cat(grads)
    grad_norm = torch.norm(grads).item()
    grad_mean = grads.mean().item()
    grad_std = grads.std().item()
    gradient_stats.append({'step':epoch,'grad_norms':grad_norm,'grad_means':grad_mean,'grad_stds':grad_std})

    total_time = fwd_time + bwd_time
    flops_per_sec = total_flops / total_time if total_time>0 else 0

    # Logging
    with open(loss_log_file,'a') as f:
        f.write(f'{epoch},{loss_res.item():.8e},{loss_bc.item():.8e},{loss_ic.item():.8e},{loss.item():.8e}\n')
    with open(flops_log_file,'a') as f:
        f.write(f'{epoch},{fwd_flops:.2e},{bwd_flops:.2e},{total_flops:.2e},{fwd_time:.6f},{bwd_time:.6f},{total_time:.6f},{flops_per_sec:.2e}\n')
    with open(grad_log_file,'a') as f:
        f.write(f'{epoch},{grad_norm:.8e},{grad_mean:.8e},{grad_std:.8e}\n')
    loss_track.append([loss_res.item(), loss_bc.item(), loss_ic.item(), loss.item()])

# =======================
# SAVE MODEL
# =======================
torch.save(model.state_dict(), f'./results/1dconvection_{args.model}_adam_fp64.pt')

# =======================
# POST-HOC TESTING & PLOTS
# =======================
res_test = torch.tensor(res_test, dtype=torch.float64, requires_grad=True).to(device)
x_test, t_test = res_test[...,0:1], res_test[...,1:2]

pred = model(x_test, t_test)
u_x_test = torch.autograd.grad(pred, x_test, torch.ones_like(pred), create_graph=True)[0]
u_t_test = torch.autograd.grad(pred, t_test, torch.ones_like(pred), create_graph=True)[0]

pred_np = pred.detach().cpu().numpy().reshape(101,101)
def u_res(x,t): return np.sin(x - BETA*t)
res_test_np, _, _, _, _ = get_data([0,2*np.pi],[0,1],101,101)
u_exact = u_res(res_test_np[:,0], res_test_np[:,1]).reshape(101,101)

# Relative Errors
rl1 = np.sum(np.abs(u_exact - pred_np))/np.sum(np.abs(u_exact))
rl2 = np.sqrt(np.sum((u_exact - pred_np)**2)/np.sum(u_exact**2))
print(f"BETA={BETA} | relative L1: {rl1:.6f}, relative L2: {rl2:.6f}")

# =======================
# Heatmaps
# =======================
plt.figure(figsize=(4,3))
plt.imshow(pred_np,extent=[0,2*np.pi,1,0],aspect='auto')
plt.title("Predicted u(x,t)")
plt.colorbar()
plt.tight_layout()
plt.savefig(f'./results/convection_{args.model}_pred.pdf')
plt.close()

plt.figure(figsize=(4,3))
plt.imshow(u_exact,extent=[0,2*np.pi,1,0],aspect='auto')
plt.title("Exact u(x,t)")
plt.colorbar()
plt.tight_layout()
plt.savefig(f'./results/convection_{args.model}_exact.pdf')
plt.close()

plt.figure(figsize=(4,3))
plt.imshow(pred_np-u_exact,extent=[0,2*np.pi,1,0],aspect='auto',cmap='coolwarm',vmin=-1,vmax=1)
plt.title("Absolute Error")
plt.colorbar()
plt.tight_layout()
plt.savefig(f'./results/convection_{args.model}_error.pdf')
plt.close()

# =======================
# Gradient plots
# =======================
grad_norms = [g['grad_norms'] for g in gradient_stats]
grad_means = [g['grad_means'] for g in gradient_stats]
grad_stds = [g['grad_stds'] for g in gradient_stats]

plt.figure(figsize=(10,6))
plt.plot(grad_norms,label='Gradient Norm')
plt.xlabel("Step")
plt.ylabel("Norm")
plt.title("Gradient Norms During Training")
plt.grid(True)
plt.savefig(f'./results/grad_norms_{args.model}.pdf')
plt.close()

plt.figure(figsize=(10,6))
plt.plot(grad_means,label='Gradient Mean')
plt.xlabel("Step")
plt.ylabel("Mean")
plt.title("Gradient Means During Training")
plt.grid(True)
plt.savefig(f'./results/grad_means_{args.model}.pdf')
plt.close()

plt.figure(figsize=(10,6))
plt.plot(grad_stds,label='Gradient Std')
plt.xlabel("Step")
plt.ylabel("Std")
plt.title("Gradient Stds During Training")
plt.grid(True)
plt.savefig(f'./results/grad_stds_{args.model}.pdf')
plt.close()

# =======================
# Loss plots
# =======================
loss_track = np.array(loss_track)
epochs_arr = np.arange(EPOCHS)
plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.plot(epochs_arr, loss_track[:,0],label="Loss Res")
plt.plot(epochs_arr, loss_track[:,1],label="Loss BC")
plt.plot(epochs_arr, loss_track[:,2],label="Loss IC")
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(epochs_arr, loss_track[:,3],label="Total Loss",color='red')
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'./results/losses_{args.model}.pdf')
plt.close()

# =======================
# FLOPs & Timing Plots
# =======================
flops_data = np.loadtxt(flops_log_file, delimiter=',')
if flops_data.shape[0] > 0:
    epochs_flops = flops_data[:,0]
    fwd_flops = flops_data[:,1]
    bwd_flops = flops_data[:,2]
    total_flops = flops_data[:,3]
    fwd_time = flops_data[:,4]
    bwd_time = flops_data[:,5]
    total_time = flops_data[:,6]
    flops_per_sec = flops_data[:,7]

    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1)
    plt.semilogy(epochs_flops,total_flops,'b',label='Total FLOPs')
    plt.semilogy(epochs_flops,fwd_flops,'r--',label='Forward FLOPs')
    plt.semilogy(epochs_flops,bwd_flops,'g--',label='Backward FLOPs')
    plt.xlabel("Epoch"); plt.ylabel("FLOPs"); plt.legend(); plt.grid(True)

    plt.subplot(2,2,2)
    plt.plot(epochs_flops, flops_per_sec,'purple',label='FLOPS')
    plt.xlabel("Epoch"); plt.ylabel("FLOPS"); plt.yscale('log'); plt.grid(True); plt.legend()

    plt.subplot(2,2,3)
    plt.plot(epochs_flops, total_time,'b',label='Total Time')
    plt.plot(epochs_flops, fwd_time,'r--',label='Forward Time')
    plt.plot(epochs_flops, bwd_time,'g--',label='Backward Time')
    plt.xlabel("Epoch"); plt.ylabel("Time (s)"); plt.legend(); plt.grid(True)

    plt.subplot(2,2,4)
    plt.semilogy(epochs_flops, np.cumsum(total_flops),'orange',label='Cumulative FLOPs')
    plt.xlabel("Epoch"); plt.ylabel("Cumulative FLOPs"); plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(f'./results/flops_{args.model}.pdf')
    plt.close()
