import time
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.optim import LBFGS
from tqdm import tqdm
import argparse
from collections import deque
from util import *
from model_dict import get_model

# =======================
# CONFIG
# =======================
TOTAL_EPOCHS = 50000
PLATEAU_WINDOW = 50
PLATEAU_EPS = 1e-3
START_DTYPE = torch.float64
SWITCH_DTYPE = torch.float32
STEP_SIZE = 1e-4
NUM_STEP = 5
BETA = 50

# =======================
# SEED
# =======================
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# =======================
# ARGPARSE
# =======================
parser = argparse.ArgumentParser('Training 1D Convection PINN')
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
    res = make_time_sequence(res, num_step=NUM_STEP, step=STEP_SIZE)
    b_left = make_time_sequence(b_left, num_step=NUM_STEP, step=STEP_SIZE)
    b_right = make_time_sequence(b_right, num_step=NUM_STEP, step=STEP_SIZE)
    b_upper = make_time_sequence(b_upper, num_step=NUM_STEP, step=STEP_SIZE)
    b_lower = make_time_sequence(b_lower, num_step=NUM_STEP, step=STEP_SIZE)
    res_test = make_time_sequence(res_test, num_step=NUM_STEP, step=STEP_SIZE)

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
        m.bias.data.fill_(0.0)

if args.model == 'KAN':
    model = get_model(args).Model(width=[2,5,5,1], grid=5, k=3, grid_eps=1.0,
                                  noise_scale_base=0.25, device=device).to(START_DTYPE).to(device)
elif args.model == 'QRes':
    model = get_model(args).Model(in_dim=2, hidden_dim=256, out_dim=1, num_layer=4).to(START_DTYPE).to(device)
    model.apply(init_weights)
elif args.model in ['PINNsFormer','PINNsFormer_Enc_Only']:
    model = get_model(args).Model(in_dim=2, hidden_dim=32, out_dim=1, num_layer=1).to(START_DTYPE).to(device)
    model.apply(init_weights)
else:
    model = get_model(args).Model(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4).to(START_DTYPE).to(device)
    model.apply(init_weights)

def make_optimizer():
    return LBFGS(model.parameters(), line_search_fn='strong_wolfe', tolerance_grad=1e-8, tolerance_change=1e-10)

optim = make_optimizer()
print(model)
print("Parameters:", sum(p.numel() for p in model.parameters()))

# =======================
# LOGGING
# =======================
os.makedirs('./results', exist_ok=True)
loss_log_path = f'./results/1dconvection_{args.model}_loss_log.txt'
flops_log_path = f'./results/1dconvection_{args.model}_flops_log.txt'
grad_log_path = f'./results/1dconvection_{args.model}_grad_log.txt'

with open(loss_log_path,'w') as f: f.write('epoch,loss_res,loss_bc,loss_ic,total_loss,precision\n')
with open(flops_log_path,'w') as f: f.write('epoch,forward_flops,backward_flops,total_flops,forward_time,backward_time,total_time,flops_per_sec,precision\n')
with open(grad_log_path,'w') as f: f.write('epoch,grad_norm,precision\n')

loss_track = []
flops_track = []
gradient_stats = []
precision_switched = False
current_dtype = START_DTYPE
loss_window = deque(maxlen=PLATEAU_WINDOW)

# =======================
# FLOPs ESTIMATION
# =======================
def estimate_flops(model, batch_size):
    flops = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            flops += (2*module.in_features*module.out_features + module.out_features) * batch_size
        elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
            flops += batch_size * getattr(module, 'out_features', 1)
    return flops

sample_batch = x_res.shape[0] * (x_res.shape[1] if len(x_res.shape)==3 else 1)
forward_flops_fp64 = estimate_flops(model, sample_batch)
backward_flops_fp64 = forward_flops_fp64*2
total_forward_fp64 = forward_flops_fp64*(5+2*2)
total_backward_fp64 = backward_flops_fp64*5
total_flops_fp64 = total_forward_fp64 + total_backward_fp64
print(f"Estimated FLOPs per epoch (FP64): {total_flops_fp64:.2e}")

# =======================
# TRAINING LOOP
# =======================
for epoch in tqdm(range(TOTAL_EPOCHS), desc="Training", ncols=100, unit="epoch"):
    timing_info = [0.0, 0.0]
    grad_norm_container = [None]

    def closure():
        optim.zero_grad()
        fwd_start = time.time()

        pred_res = model(x_res, t_res)
        pred_left = model(x_left, t_left)
        pred_right = model(x_right, t_right)
        pred_upper = model(x_upper, t_upper)
        pred_lower = model(x_lower, t_lower)

        u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=torch.ones_like(pred_res),
                                  retain_graph=True, create_graph=True)[0]
        u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res),
                                  retain_graph=True, create_graph=True)[0]

        loss_res = torch.mean((u_t + BETA * u_x)**2)
        loss_bc = torch.mean((pred_upper - pred_lower)**2)
        loss_ic = torch.mean((pred_left[:,0] - torch.sin(x_left[:,0]))**2)

        fwd_end = time.time()
        timing_info[0] = fwd_end - fwd_start

        loss = loss_res + loss_bc + loss_ic

        bwd_start = time.time()
        loss.backward()
        bwd_end = time.time()
        timing_info[1] = bwd_end - bwd_start

        # gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None: total_norm += p.grad.data.norm(2).item()**2
        grad_norm_container[0] = total_norm**0.5

        loss_track.append([loss_res.item(), loss_bc.item(), loss_ic.item(), loss.item()])
        return loss

    optim.step(closure)

    loss_res_v, loss_bc_v, loss_ic_v, total_loss_v = loss_track[-1]
    loss_window.append(total_loss_v)
    grad_norm = grad_norm_container[0]

    # Precision switch
    if not precision_switched and len(loss_window)==PLATEAU_WINDOW:
        L_max, L_min = max(loss_window), min(loss_window)
        delta = (L_max-L_min)/max(L_min,1e-12)
        if delta<PLATEAU_EPS:
            precision_switched=True
            current_dtype = SWITCH_DTYPE
            print(f"\n🔁 PRECISION SWITCH TRIGGERED at epoch {epoch+1}, Δ={delta:.3e}, loss={total_loss_v:.3e}\n")
            model = model.to(SWITCH_DTYPE)
            for tensor_name in ['x_res','t_res','x_left','t_left','x_right','t_right','x_upper','t_upper','x_lower','t_lower']:
                globals()[tensor_name] = globals()[tensor_name].to(SWITCH_DTYPE)
            optim = make_optimizer()

    # Logging FLOPs
    epoch_total_time = sum(timing_info)
    flops_track.append([total_forward_fp64, total_backward_fp64, total_flops_fp64, timing_info[0], timing_info[1], epoch_total_time, total_flops_fp64/epoch_total_time if epoch_total_time>0 else 0.0])

    # Write logs
    with open(loss_log_path,'a') as f:
        f.write(f"{epoch+1},{loss_res_v:.8e},{loss_bc_v:.8e},{loss_ic_v:.8e},{total_loss_v:.8e},{current_dtype}\n")
    if grad_norm is not None:
        with open(grad_log_path,'a') as f:
            f.write(f"{epoch+1},{grad_norm:.8e},{current_dtype}\n")

# =======================
# SAVE MODEL
# =======================
torch.save(model.state_dict(), f'./results/1dconvection_{args.model}_point.pt')

# =======================
# TEST AND PLOTS
# =======================
res_test = torch.tensor(res_test, dtype=START_DTYPE, requires_grad=True).to(device)
x_test, t_test = res_test[...,0:1], res_test[...,1:2]

with torch.no_grad():
    pred = model(x_test, t_test)[:,0:1].cpu().numpy()

pred = pred.reshape(101,101)

def u_res(x,t): return np.sin(x - BETA*t)
res_test, _, _, _, _ = get_data([0,2*np.pi],[0,1],101,101)
u = u_res(res_test[:,0], res_test[:,1]).reshape(101,101)

rl1 = np.sum(np.abs(u-pred))/np.sum(np.abs(u))
rl2 = np.sqrt(np.sum((u-pred)**2)/np.sum(u**2))
print(f"BETA: {BETA}, relative L1: {rl1:.4f}, relative L2: {rl2:.4f}")

# Prediction plots
plt.figure(figsize=(4,3))
plt.imshow(pred, extent=[0,2*np.pi,1,0], aspect='auto')
plt.xlabel('x'); plt.ylabel('t'); plt.title('Predicted u(x,t)'); plt.colorbar()
plt.tight_layout(); plt.savefig(f'./results/convection_{args.model}_pred.pdf', bbox_inches='tight')

plt.figure(figsize=(4,3))
plt.imshow(u, extent=[0,2*np.pi,1,0], aspect='auto')
plt.xlabel('x'); plt.ylabel('t'); plt.title('Exact u(x,t)'); plt.colorbar()
plt.tight_layout(); plt.savefig(f'./results/convection_exact.pdf', bbox_inches='tight')

plt.figure(figsize=(4,3))
plt.imshow(pred-u, extent=[0,2*np.pi,1,0], aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
plt.xlabel('x'); plt.ylabel('t'); plt.title('Absolute Error'); plt.colorbar()
plt.tight_layout(); plt.savefig(f'./results/convection_{args.model}_error.pdf', bbox_inches='tight')
