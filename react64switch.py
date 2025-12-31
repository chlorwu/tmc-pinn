import time
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS
from tqdm import tqdm
import argparse
import numpy as np
from collections import deque
from util import *
from model_dict import get_model

# =======================
# CONFIG
# =======================
TOTAL_EPOCHS = 2000
PLATEAU_WINDOW = 50       # number of epochs to monitor for plateau
PLATEAU_EPS = 1e-4        # relative change threshold to detect plateau
START_DTYPE = torch.float64
SWITCH_DTYPE = torch.float32
STEP_SIZE = 1e-4
NUM_STEP = 5
# =======================

seed = 1
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

parser = argparse.ArgumentParser('Training Point Optimization')
parser.add_argument('--model', type=str, default='PINN')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
device = args.device

# =======================
# DATA
# =======================
res, b_left, b_right, b_upper, b_lower = get_data([0, 2 * np.pi], [0, 1], 101, 101)
res_test, _, _, _, _ = get_data([0, 2 * np.pi], [0, 1], 101, 101)

if args.model in ['PINNsFormer', 'PINNMamba']:
    res = make_time_sequence(res, num_step=10, step=STEP_SIZE)
    b_left = make_time_sequence(b_left, num_step=NUM_STEP, step=STEP_SIZE)
    b_right = make_time_sequence(b_right, num_step=NUM_STEP, step=STEP_SIZE)
    b_upper = make_time_sequence(b_upper, num_step=NUM_STEP, step=STEP_SIZE)
    b_lower = make_time_sequence(b_lower, num_step=NUM_STEP, step=STEP_SIZE)

res = torch.tensor(res, dtype=START_DTYPE, requires_grad=True).to(device)
b_left = torch.tensor(b_left, dtype=START_DTYPE, requires_grad=True).to(device)
b_right = torch.tensor(b_right, dtype=START_DTYPE, requires_grad=True).to(device)
b_upper = torch.tensor(b_upper, dtype=START_DTYPE, requires_grad=True).to(device)
b_lower = torch.tensor(b_lower, dtype=START_DTYPE, requires_grad=True).to(device)

x_res, t_res = res[:, ..., 0:1], res[:, ..., 1:2]
x_left, t_left = b_left[:, ..., 0:1], b_left[:, ..., 1:2]
x_right, t_right = b_right[:, ..., 0:1], b_right[:, ..., 1:2]
x_upper, t_upper = b_upper[:, ..., 0:1], b_upper[:, ..., 1:2]
x_lower, t_lower = b_lower[:, ..., 0:1], b_lower[:, ..., 1:2]

# =======================
# MODEL
# =======================
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

if args.model == 'KAN':
    model = get_model(args).Model(width=[2,5,1], grid=5, k=3, grid_eps=1.0,
                                  noise_scale_base=0.25, device=device).to(START_DTYPE).to(device)
elif args.model == 'QRes':
    model = get_model(args).Model(in_dim=2, hidden_dim=256, out_dim=1, num_layer=4).to(START_DTYPE).to(device)
    model.apply(init_weights)
elif args.model in ['PINNsFormer', 'PINNsFormer_Enc_Only']:
    model = get_model(args).Model(in_dim=2, hidden_dim=32, out_dim=1, num_layer=1).to(START_DTYPE).to(device)
    model.apply(init_weights)
else:
    model = get_model(args).Model(in_dim=2, hidden_dim=1024, out_dim=1, num_layer=6).to(START_DTYPE).to(device)
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
log_file_path = f'./results/1dreaction_{args.model}_loss_log.txt'
flops_log_file_path = f'./results/1dreaction_{args.model}_flops_log.txt'
gradient_log_file_path = f'./results/1dreaction_{args.model}_gradient_log.txt'

with open(log_file_path, 'w') as f:
    f.write('epoch,loss_res,loss_bc,loss_ic,total_loss,precision\n')
with open(flops_log_file_path, 'w') as f:
    f.write('epoch,forward_flops,backward_flops,total_flops,forward_time,backward_time,total_time,flops_per_sec,precision\n')
with open(gradient_log_file_path, 'w') as f:
    f.write('epoch,gradient_norm,precision\n')

loss_track = []
flops_track = []
precision_switched = False
current_dtype = START_DTYPE
loss_window = deque(maxlen=PLATEAU_WINDOW)

# =======================
# FLOPs ESTIMATION FUNCTION
# =======================
def estimate_flops(model, input_shape):
    flops = 0
    batch_size = input_shape[0]
    for module in model.modules():
        if isinstance(module, nn.Linear):
            flops += (2 * module.in_features * module.out_features + module.out_features) * batch_size
        elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
            flops += batch_size * getattr(module, 'out_features', 1)
    return flops

# =======================
# CALCULATE FLOPs ONCE FOR FP64
# =======================
sample_batch_size = x_res.shape[0] * (x_res.shape[1] if len(x_res.shape)==3 else 1)
forward_flops_per_pass_fp64 = estimate_flops(model, (sample_batch_size,))
backward_flops_per_pass_fp64 = forward_flops_per_pass_fp64 * 2
num_forward_passes = 5
num_grad_computations = 2
total_forward_flops_fp64 = forward_flops_per_pass_fp64 * (num_forward_passes + num_grad_computations * 2)
total_backward_flops_fp64 = backward_flops_per_pass_fp64 * num_forward_passes
total_flops_per_epoch_fp64 = total_forward_flops_fp64 + total_backward_flops_fp64

print(f"Estimated FLOPs per epoch (FP64): {total_flops_per_epoch_fp64:.2e}")
print(f"  Forward FLOPs: {total_forward_flops_fp64:.2e}")
print(f"  Backward FLOPs: {total_backward_flops_fp64:.2e}")

# =======================
# CALCULATE FLOPs ONCE FOR FP32 (after precision switch)
# =======================
total_forward_flops_fp32 = None
total_backward_flops_fp32 = None
total_flops_per_epoch_fp32 = None

# =======================
# TRAINING LOOP
# =======================
for epoch in tqdm(range(TOTAL_EPOCHS), desc="Training"):
    timing_info = [0.0, 0.0]
    grad_norm = None

    def closure():
        optim.zero_grad()
        fwd_start = time.time()

        pred_res = model(x_res, t_res)
        pred_left = model(x_left, t_left)
        pred_right = model(x_right, t_right)
        pred_upper = model(x_upper, t_upper)
        pred_lower = model(x_lower, t_lower)

        u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
        u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]

        loss_res = torch.mean((u_t - 5 * pred_res * (1 - pred_res)) ** 2)
        loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
        loss_ic = torch.mean((pred_left[:, 0] - torch.exp(-(x_left[:,0]-torch.pi)**2/(2*(torch.pi/4)**2)))**2)

        fwd_end = time.time()
        timing_info[0] = fwd_end - fwd_start
        loss = loss_res + loss_bc + loss_ic

        bwd_start = time.time()
        loss.backward()
        bwd_end = time.time()
        timing_info[1] = bwd_end - bwd_start

        # Compute gradient norm
        grad_norm_list = []
        for p in model.parameters():
            if p.grad is not None:
                grad_norm_list.append(p.grad.data.norm(2).item() ** 2)
        grad_norm_val = sum(grad_norm_list) ** 0.5

        loss_track.append([loss_res.item(), loss_bc.item(), loss_ic.item(), loss.item()])
        return loss, grad_norm_val

    loss, grad_norm = closure()
    optim.step(closure)
    loss_res_v, loss_bc_v, loss_ic_v, total_loss_v = loss_track[-1]
    loss_window.append(total_loss_v)

    # =======================
    # PLATEAU-BASED PRECISION SWITCH
    # =======================
    if not precision_switched and len(loss_window) == PLATEAU_WINDOW:
        L_max, L_min = max(loss_window), min(loss_window)
        delta = (L_max - L_min)/max(L_min,1e-12)
        if delta < PLATEAU_EPS:
            precision_switched = True
            current_dtype = SWITCH_DTYPE
            print("\n" + "="*60)
            print(f"🔁 PRECISION SWITCH TRIGGERED at epoch {epoch+1}, Δ={delta:.3e}, loss={total_loss_v:.3e}")
            print("="*60 + "\n")
            
            # Convert model and data
            model = model.to(SWITCH_DTYPE)
            for tensor_name in ['x_res','t_res','x_left','t_left','x_right','t_right','x_upper','t_upper','x_lower','t_lower']:
                globals()[tensor_name] = globals()[tensor_name].to(SWITCH_DTYPE)
            optim = make_optimizer()
            
            # =======================
            # CALCULATE FLOPs ONCE FOR FP32 (after precision switch)
            # =======================
            forward_flops_per_pass_fp32 = estimate_flops(model, (sample_batch_size,))
            backward_flops_per_pass_fp32 = forward_flops_per_pass_fp32 * 2
            total_forward_flops_fp32 = forward_flops_per_pass_fp32 * (num_forward_passes + num_grad_computations * 2)
            total_backward_flops_fp32 = backward_flops_per_pass_fp32 * num_forward_passes
            total_flops_per_epoch_fp32 = total_forward_flops_fp32 + total_backward_flops_fp32
            
            print(f"Estimated FLOPs per epoch (FP32): {total_flops_per_epoch_fp32:.2e}")
            print(f"  Forward FLOPs: {total_forward_flops_fp32:.2e}")
            print(f"  Backward FLOPs: {total_backward_flops_fp32:.2e}")

    # =======================
    # LOGGING
    # =======================
    epoch_total_time = sum(timing_info)
    
    # Use appropriate FLOPs based on current precision
    if current_dtype == torch.float32 and total_flops_per_epoch_fp32 is not None:
        total_forward_flops = total_forward_flops_fp32
        total_backward_flops = total_backward_flops_fp32
        total_flops_per_epoch = total_flops_per_epoch_fp32
        precision_str = 'fp32'
    else:
        total_forward_flops = total_forward_flops_fp64
        total_backward_flops = total_backward_flops_fp64
        total_flops_per_epoch = total_flops_per_epoch_fp64
        precision_str = 'fp64'
    
    flops_per_sec = total_flops_per_epoch / epoch_total_time if epoch_total_time>0 else 0.0
    flops_track.append([total_forward_flops, total_backward_flops, total_flops_per_epoch,
                        timing_info[0], timing_info[1], epoch_total_time, flops_per_sec])

    with open(log_file_path,'a') as f:
        f.write(f"{epoch+1},{loss_res_v:.8e},{loss_bc_v:.8e},{loss_ic_v:.8e},{total_loss_v:.8e},{precision_str}\n")
    
    with open(flops_log_file_path,'a') as f:
        f.write(f"{epoch+1},{total_forward_flops:.2e},{total_backward_flops:.2e},{total_flops_per_epoch:.2e},"
                f"{timing_info[0]:.6f},{timing_info[1]:.6f},{epoch_total_time:.6f},{flops_per_sec:.2e},{precision_str}\n")
    
    # Log gradient norm
    if grad_norm is not None:
        with open(gradient_log_file_path,'a') as f:
            f.write(f"{epoch+1},{grad_norm:.8e},{precision_str}\n")

# =======================
# SAVE MODEL AND RESULTS
# =======================
torch.save(model.state_dict(), f'./results/1dreaction_{args.model}_point.pt')
