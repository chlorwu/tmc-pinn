import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.optim import LBFGS, Adam
from tqdm import tqdm
import os
import argparse
from util import *
from model_dict import get_model

# --------------------------
# Seed
# --------------------------
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# --------------------------
# Hyperparameters
# --------------------------
step_size = 1e-4
num_step = 5
beta = 50
switch_epoch = 2500  # Epoch to switch precision from float32 to float64
max_epochs = 50000

# --------------------------
# Argument parser
# --------------------------
parser = argparse.ArgumentParser('Training Point Optimization')
parser.add_argument('--model', type=str, default='PINN')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
device = args.device

# --------------------------
# Data
# --------------------------
res, b_left, b_right, b_upper, b_lower = get_data([0, 2 * np.pi], [0, 1], 401, 401)
res_test, b_left_test, _, _, _ = get_data([0, 2 * np.pi], [0, 1], 101, 101)

if args.model in ['PINNsFormer', 'PINNMamba']:
    res = make_time_sequence(res, num_step=num_step, step=step_size)
    b_left = make_time_sequence(b_left, num_step=num_step, step=step_size)
    b_right = make_time_sequence(b_right, num_step=num_step, step=step_size)
    b_upper = make_time_sequence(b_upper, num_step=num_step, step=step_size)
    b_lower = make_time_sequence(b_lower, num_step=num_step, step=step_size)
    res_test = make_time_sequence(res_test, num_step=num_step, step=step_size)

# Convert to tensors with requires_grad=True
res = torch.tensor(res, dtype=torch.float32, requires_grad=True).to(device)
b_left = torch.tensor(b_left, dtype=torch.float32, requires_grad=True).to(device)
b_right = torch.tensor(b_right, dtype=torch.float32, requires_grad=True).to(device)
b_upper = torch.tensor(b_upper, dtype=torch.float32, requires_grad=True).to(device)
b_lower = torch.tensor(b_lower, dtype=torch.float32, requires_grad=True).to(device)

x_res, t_res = res[..., 0:1], res[..., 1:2]
x_left, t_left = b_left[..., 0:1], b_left[..., 1:2]
x_right, t_right = b_right[..., 0:1], b_right[..., 1:2]
x_upper, t_upper = b_upper[..., 0:1], b_upper[..., 1:2]
x_lower, t_lower = b_lower[..., 0:1], b_lower[..., 1:2]

# --------------------------
# Model initialization
# --------------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

if args.model == 'KAN':
    model = get_model(args).Model(width=[2, 5, 5, 1], grid=5, k=3, grid_eps=1.0, noise_scale_base=0.25, device=device).to(device)
elif args.model == 'QRes':
    model = get_model(args).Model(in_dim=2, hidden_dim=256, out_dim=1, num_layer=4).to(device)
    model.apply(init_weights)
elif args.model in ['PINNsFormer', 'PINNsFormer_Enc_Only']:
    model = get_model(args).Model(in_dim=2, hidden_dim=32, out_dim=1, num_layer=1).to(device)
    model.apply(init_weights)
else:
    model = get_model(args).Model(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4).to(device)
    model.apply(init_weights)

# --------------------------
# Optimizer
# --------------------------
optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe', tolerance_grad=1e-8, tolerance_change=1e-10)

# --------------------------
# Logging setup
# --------------------------
os.makedirs('./results/', exist_ok=True)
loss_log_file = f'./results/1dconvection_{args.model}_{num_step}_{step_size}_{beta}_loss_log.txt'
with open(loss_log_file, 'w') as f:
    f.write('epoch,loss_res,loss_bc,loss_ic,total_loss\n')

flops_log_file = f'./results/1dconvection_{args.model}_{num_step}_{step_size}_{beta}_flops_log.txt'
with open(flops_log_file, 'w') as f:
    f.write('epoch,forward_flops,backward_flops,total_flops,forward_time,backward_time,total_time,flops_per_sec\n')

# --------------------------
# FLOPs estimation
# --------------------------
def estimate_flops(model, input_shape):
    flops = 0
    batch_size = input_shape[0]
    for module in model.modules():
        if isinstance(module, nn.Linear):
            flops += (2 * module.in_features * module.out_features + module.out_features) * batch_size
        elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
            flops += batch_size * getattr(module, 'out_features', 1)
    return flops

if len(x_res.shape) >= 2:
    sample_batch_size = x_res.shape[0] * (x_res.shape[1] if len(x_res.shape) == 3 else 1)
else:
    sample_batch_size = 1

forward_flops_per_pass = estimate_flops(model, (sample_batch_size,))
backward_flops_per_pass = 2 * forward_flops_per_pass

# --------------------------
# Training loop
# --------------------------
loss_track = []
gradient_stats = []
flops_track = []

for epoch in tqdm(range(max_epochs)):

    # --------------------------
    # Precision switch
    # --------------------------
    if epoch == switch_epoch:
        model = model.to(torch.float64)
        x_res, t_res = x_res.to(torch.float64), t_res.to(torch.float64)
        x_left, t_left = x_left.to(torch.float64), t_left.to(torch.float64)
        x_right, t_right = x_right.to(torch.float64), t_right.to(torch.float64)
        x_upper, t_upper = x_upper.to(torch.float64), t_upper.to(torch.float64)
        x_lower, t_lower = x_lower.to(torch.float64), t_lower.to(torch.float64)

    timing_info = [0.0, 0.0]

    def closure():
        forward_start = time.time()

        pred_res = model(x_res, t_res)
        pred_left = model(x_left, t_left)
        pred_right = model(x_right, t_right)
        pred_upper = model(x_upper, t_upper)
        pred_lower = model(x_lower, t_lower)

        u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=torch.ones_like(pred_res),
                                  retain_graph=True, create_graph=True)[0]
        u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res),
                                  retain_graph=True, create_graph=True)[0]

        loss_res = torch.mean((u_t + beta * u_x) ** 2)
        loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
        loss_ic = torch.mean((pred_left[:, 0] - torch.sin(x_left[:, 0])) ** 2)

        forward_end = time.time()
        timing_info[0] = forward_end - forward_start

        loss = loss_res + loss_bc + loss_ic
        optim.zero_grad()

        backward_start = time.time()
        loss.backward()
        backward_end = time.time()
        timing_info[1] = backward_end - backward_start

        loss_track.append([loss_res.item(), loss_bc.item(), loss_ic.item()])
        return loss

    optim.step(closure)

    total_time = sum(timing_info)
    # Compute FLOPs dynamically
    num_forward_passes = 5
    num_grad_computations = 2
    forward_flops = forward_flops_per_pass * num_forward_passes
    grad_flops = forward_flops_per_pass * num_grad_computations * 2
    total_forward_flops = forward_flops + grad_flops
    total_backward_flops = backward_flops_per_pass * num_forward_passes
    total_flops = total_forward_flops + total_backward_flops
    flops_per_sec = total_flops / total_time if total_time > 0 else 0.0
    flops_track.append([total_forward_flops, total_backward_flops, total_flops, timing_info[0], timing_info[1], total_time, flops_per_sec])

    # Logging
    with open(loss_log_file, 'a') as f:
        lres, lbc, lic = loss_track[-1]
        f.write(f'{epoch},{lres:.10e},{lbc:.10e},{lic:.10e},{lres + lbc + lic:.10e}\n')

    with open(flops_log_file, 'a') as f:
        f.write(f'{epoch+1},{total_forward_flops:.2e},{total_backward_flops:.2e},{total_flops:.2e},'
                f'{timing_info[0]:.6f},{timing_info[1]:.6f},{total_time:.6f},{flops_per_sec:.2e}\n')

    # Gradient stats after 50 epochs
    if epoch > 50:
        grad_norms, grad_means, grad_stds = [], [], []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
                grad_means.append(param.grad.mean().item())
                grad_stds.append(param.grad.std().item())
        gradient_stats.append({'step': epoch, 'grad_norms': grad_norms, 'grad_means': grad_means, 'grad_stds': grad_stds})

# --------------------------
# Save model
# --------------------------
torch.save(model.state_dict(), f'./results/1dconvection_{args.model}_{num_step}_{step_size}.pt')

# --------------------------
# Evaluation on test data
# --------------------------
res_test = torch.tensor(res_test, dtype=torch.float64, requires_grad=True).to(device)
x_test, t_test = res_test[..., 0:1], res_test[..., 1:2]

pred = model(x_test, t_test)
pred = pred.cpu().detach().numpy().reshape(101, 101)

def u_res(x, t):
    return np.sin(x - beta * t)

res_test, _, _, _, _ = get_data([0, 2 * np.pi], [0, 1], 101, 101)
u = u_res(res_test[:, 0], res_test[:, 1]).reshape(101, 101)

rl1 = np.sum(np.abs(u - pred)) / np.sum(np.abs(u))
rl2 = np.sqrt(np.sum((u - pred) ** 2) / np.sum(u ** 2))

print(beta)
print('relative L1 error: {:4f}'.format(rl1))
print('relative L2 error: {:4f}'.format(rl2))

# --------------------------
# Plotting everything
# --------------------------
# Loss
loss_data = np.loadtxt(loss_log_file, delimiter=',', skiprows=1)
epochs = loss_data[:, 0]
loss_res = loss_data[:, 1]
loss_bc = loss_data[:, 2]
loss_ic = loss_data[:, 3]
total_loss = loss_data[:, 4]

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(epochs, loss_res, label='Loss Res', alpha=0.7)
plt.plot(epochs, loss_bc, label='Loss BC', alpha=0.7)
plt.plot(epochs, loss_ic, label='Loss IC', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Individual Loss Components')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(2, 1, 2)
plt.plot(epochs, total_loss, label='Total Loss', color='red', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Total Loss vs Epoch')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig(f'./results/1d_convection_{args.model}_{beta}_loss_vs_epoch.pdf')
plt.close()

# Gradient plots
grad_norms_history = [s['grad_norms'] for s in gradient_stats]
grad_means_history = [s['grad_means'] for s in gradient_stats]
grad_stds_history = [s['grad_stds'] for s in gradient_stats]

plt.figure(figsize=(10, 6))
for i in range(len(grad_norms_history[0])):
    plt.plot([s[i] for s in grad_norms_history], label=f'Layer {i+1}')
plt.xlabel('Training Step')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norms During Training')
plt.legend()
plt.grid()
plt.savefig(f'./results/1d_convection_{args.model}_{beta}_gradient_norms.pdf')
plt.close()

plt.figure(figsize=(10, 6))
for i in range(len(grad_means_history[0])):
    plt.plot([s[i] for s in grad_means_history], label=f'Layer {i+1}')
plt.xlabel('Training Step')
plt.ylabel('Gradient Mean')
plt.title('Gradient Means During Training')
plt.legend()
plt.grid()
plt.savefig(f'./results/1d_convection_{args.model}_{beta}_gradient_means.pdf')
plt.close()

plt.figure(figsize=(10, 6))
for i in range(len(grad_stds_history[0])):
    plt.plot([s[i] for s in grad_stds_history], label=f'Layer {i+1}')
plt.xlabel('Training Step')
plt.ylabel('Gradient Std')
plt.title('Gradient Stds During Training')
plt.legend()
plt.grid()
plt.savefig(f'./results/1d_convection_{args.model}_{beta}_gradient_stds.pdf')
plt.close()

# Prediction heatmaps
plt.figure(figsize=(4, 3))
plt.imshow(pred, extent=[0, 2*np.pi, 1, 0], aspect='auto')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Predicted u(x,t)')
plt.colorbar()
plt.tight_layout()
plt.savefig(f'./results/convection_{args.model}_{num_step}_{step_size}_{beta}_pred.pdf')
plt.close()

plt.figure(figsize=(4, 3))
plt.imshow(u, extent=[0, 2*np.pi, 1, 0], aspect='auto')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Exact u(x,t)')
plt.colorbar()
plt.tight_layout()
plt.savefig(f'./results/convection_exact_{beta}.pdf')
plt.close()

plt.figure(figsize=(4, 3))
plt.imshow(pred - u, extent=[0, 2*np.pi, 1, 0], aspect='auto', cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Absolute Error')
plt.colorbar()
plt.tight_layout()
plt.savefig(f'./results/convection_{args.model}_{num_step}_{step_size}_{beta}_error.pdf')
plt.close()
