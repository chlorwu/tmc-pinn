import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# =====================
# Argument parsing
# =====================
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--model', type=str, default='MLP')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

# =====================
# Seed control
# =====================
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# =====================
# FP64 enforcement
# =====================
torch.set_default_dtype(torch.float64)

# =====================
# PDE parameters
# =====================
beta = 50.0
num_step = 5
step_size = 1e-4

# =====================
# Result directory
# =====================
os.makedirs("./results", exist_ok=True)

loss_log_file = f'./results/1dconvection_loss_log.txt'
flops_log_file_path = f'./results/1dconvection_flops_log.txt'

with open(loss_log_file, 'w') as f:
    f.write('epoch,loss_res,loss_bc,loss_ic,total_loss\n')

with open(flops_log_file_path, 'w') as f:
    f.write('epoch,forward_flops,backward_flops,total_flops,forward_time,backward_time,total_time,flops_per_sec\n')

# =====================
# Data generation
# =====================
def get_data(nx, nt):
    x = np.linspace(0, 2*np.pi, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)
    XT = np.stack([X.flatten(), T.flatten()], axis=1)
    return XT, X, T

XT, Xg, Tg = get_data(101, 101)
XT = torch.tensor(XT, requires_grad=True).to(args.device)

# =====================
# Exact solution
# =====================
def u_exact(x, t):
    return np.sin(x - beta * t)

# =====================
# Model
# =====================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.net(x)

model = MLP().to(args.device)

# =====================
# Optimizer
# =====================
optim = torch.optim.Adam(model.parameters(), lr=args.lr)

# =====================
# FLOPs estimation
# =====================
def estimate_flops(model, batch):
    flops = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            flops += 2 * m.in_features * m.out_features * batch
    return flops

forward_flops_per_pass = estimate_flops(model, XT.shape[0])
backward_flops_per_pass = forward_flops_per_pass * 2

# =====================
# Training
# =====================
loss_track = []
gradient_stats = []
flops_track = []

for epoch in tqdm(range(args.epochs)):
    t0 = time.time()

    optim.zero_grad()

    fwd_start = time.time()
    u = model(XT)
    x = XT[:, 0:1]
    t = XT[:, 1:2]

    u_x = torch.autograd.grad(u, x, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), retain_graph=True, create_graph=True)[0]

    loss_res = torch.mean((u_t + beta * u_x) ** 2)
    loss_ic = torch.mean((model(torch.cat([x, torch.zeros_like(t)], 1)) - torch.sin(x)) ** 2)
    loss_bc = torch.tensor(0.0, device=args.device)

    loss = loss_res + loss_ic + loss_bc
    fwd_time = time.time() - fwd_start

    bwd_start = time.time()
    loss.backward()
    optim.step()
    bwd_time = time.time() - bwd_start

    total_time = time.time() - t0

    # =====================
    # Logging
    # =====================
    loss_track.append([loss_res.item(), loss_bc.item(), loss_ic.item()])
    with open(loss_log_file, 'a') as f:
        f.write(f"{epoch},{loss_res.item():.6e},{loss_bc.item():.6e},{loss_ic.item():.6e},{loss.item():.6e}\n")

    total_flops = forward_flops_per_pass + backward_flops_per_pass
    flops_per_sec = total_flops / total_time

    with open(flops_log_file_path, 'a') as f:
        f.write(f"{epoch},{forward_flops_per_pass:.2e},{backward_flops_per_pass:.2e},{total_flops:.2e},"
                f"{fwd_time:.6f},{bwd_time:.6f},{total_time:.6f},{flops_per_sec:.2e}\n")

    if epoch > 50:
        norms, means, stds = [], [], []
        for p in model.parameters():
            if p.grad is not None:
                norms.append(torch.norm(p.grad).item())
                means.append(p.grad.mean().item())
                stds.append(p.grad.std().item())
        gradient_stats.append({
            'grad_norms': norms,
            'grad_means': means,
            'grad_stds': stds
        })

# =====================
# Evaluation
# =====================
with torch.no_grad():
    pred = model(XT).cpu().numpy().reshape(101, 101)

u = u_exact(Xg, Tg)

rl1 = np.sum(np.abs(u - pred)) / np.sum(np.abs(u))
rl2 = np.sqrt(np.sum((u - pred)**2) / np.sum(u**2))

print(beta)
print(f"relative L1 error: {rl1:.6f}")
print(f"relative L2 error: {rl2:.6f}")

# =====================
# Plots (ALL requested)
# =====================
plt.figure(figsize=(4,3))
plt.imshow(pred, extent=[0,2*np.pi,1,0], aspect='auto')
plt.colorbar()
plt.savefig(f'./results/convection_pred.pdf')

plt.figure(figsize=(4,3))
plt.imshow(u, extent=[0,2*np.pi,1,0], aspect='auto')
plt.colorbar()
plt.savefig(f'./results/convection_exact.pdf')

plt.figure(figsize=(4,3))
plt.imshow(pred-u, extent=[0,2*np.pi,1,0], cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.savefig(f'./results/convection_error.pdf')

grad_norms_history = [g['grad_norms'] for g in gradient_stats]
grad_means_history = [g['grad_means'] for g in gradient_stats]
grad_stds_history = [g['grad_stds'] for g in gradient_stats]

def plot_grad(hist, name):
    plt.figure(figsize=(10,6))
    for i in range(len(hist[0])):
        plt.plot([h[i] for h in hist])
    plt.savefig(f'./results/{name}.pdf')
    plt.close()

plot_grad(grad_norms_history, 'gradient_norms')
plot_grad(grad_means_history, 'gradient_means')
plot_grad(grad_stds_history, 'gradient_stds')
