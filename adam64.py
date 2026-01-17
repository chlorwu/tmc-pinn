import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm

# =====================================================
# ARGUMENTS
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--epochs', type=int, default=50000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--beta', type=float, default=50.0)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

device = args.device
EPOCHS = args.epochs
beta = args.beta

# =====================================================
# SEED CONTROL
# =====================================================
torch.set_default_dtype(torch.float64)
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# =====================================================
# RESULT DIRECTORY
# =====================================================
os.makedirs('./results', exist_ok=True)
loss_log_file = './results/loss_log.csv'
flops_log_file = './results/flops_log.csv'
grad_log_file = './results/grad_log.csv'

with open(loss_log_file, 'w') as f:
    f.write('epoch,loss_res,loss_ic,total_loss\n')
with open(flops_log_file, 'w') as f:
    f.write('epoch,fwd_time,bwd_time,total_time,flops,flops_per_sec\n')
with open(grad_log_file, 'w') as f:
    f.write('epoch,grad_norm,grad_mean,grad_std\n')

# =====================================================
# DATA (1D CONVECTION)
# =====================================================
Nx, Nt = 256, 100
x = np.linspace(0, 2*np.pi, Nx)
t = np.linspace(0, 1, Nt)
X, T = np.meshgrid(x, t)
XT = np.stack([X.flatten(), T.flatten()], axis=1)

XT = torch.tensor(XT, requires_grad=True).to(device)

# Exact solution (transport)
u_exact = np.sin(X - beta * T)
u_exact = torch.tensor(u_exact.flatten()).to(device)

# =====================================================
# MODEL
# =====================================================
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

    def forward(self, xt):
        return self.net(xt)

model = PINN().to(device)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()

model.apply(init_weights)

optimizer = Adam(model.parameters(), lr=args.lr)

# =====================================================
# FLOPS ESTIMATION
# =====================================================
def estimate_flops(model, batch):
    flops = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            flops += 2 * m.in_features * m.out_features * batch
    return flops

BATCH = XT.shape[0]
FLOPS = estimate_flops(model, BATCH) * 3  # fwd + bwd

# =====================================================
# TRAINING LOOP
# =====================================================
loss_history = []
grad_stats = []

for epoch in tqdm(range(EPOCHS), ncols=100):

    optimizer.zero_grad()

    t0 = time.time()
    u = model(XT)
    fwd_time = time.time() - t0

    grads = torch.autograd.grad(
        outputs=u,
        inputs=XT,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]

    loss_res = torch.mean((u_t + beta * u_x) ** 2)

    x0 = XT[:, 0:1]
    t0 = torch.zeros_like(x0)
    xt0 = torch.cat([x0, t0], dim=1)
    loss_ic = torch.mean((model(xt0) - torch.sin(x0)) ** 2)

    loss = loss_res + loss_ic

    t1 = time.time()
    loss.backward()
    bwd_time = time.time() - t1

    optimizer.step()

    # Gradient stats
    grads_all = torch.cat([p.grad.flatten() for p in model.parameters()])
    grad_norm = grads_all.norm().item()
    grad_mean = grads_all.mean().item()
    grad_std = grads_all.std().item()

    total_time = fwd_time + bwd_time
    flops_sec = FLOPS / total_time

    # Logging
    with open(loss_log_file, 'a') as f:
        f.write(f"{epoch},{loss_res.item()},{loss_ic.item()},{loss.item()}\n")

    with open(flops_log_file, 'a') as f:
        f.write(f"{epoch},{fwd_time},{bwd_time},{total_time},{FLOPS},{flops_sec}\n")

    with open(grad_log_file, 'a') as f:
        f.write(f"{epoch},{grad_norm},{grad_mean},{grad_std}\n")

    loss_history.append(loss.item())
    grad_stats.append((grad_norm, grad_mean, grad_std))

# =====================================================
# EVALUATION
# =====================================================
with torch.no_grad():
    pred = model(XT).cpu().numpy().reshape(Nt, Nx)
    u_true = u_exact.cpu().numpy().reshape(Nt, Nx)

rl1 = np.sum(np.abs(u_true - pred)) / np.sum(np.abs(u_true))
rl2 = np.sqrt(np.sum((u_true - pred)**2) / np.sum(u_true**2))

print(beta)
print(f"relative L1 error: {rl1:.6f}")
print(f"relative L2 error: {rl2:.6f}")

# =====================================================
# PLOTS
# =====================================================
plt.figure()
plt.imshow(pred, extent=[0,2*np.pi,1,0], aspect='auto')
plt.colorbar()
plt.title("Predicted u(x,t)")
plt.savefig('./results/pred.pdf')
plt.close()

plt.figure()
plt.imshow(u_true, extent=[0,2*np.pi,1,0], aspect='auto')
plt.colorbar()
plt.title("Exact u(x,t)")
plt.savefig('./results/exact.pdf')
plt.close()

plt.figure()
plt.imshow(pred - u_true, extent=[0,2*np.pi,1,0], aspect='auto', cmap='coolwarm')
plt.colorbar()
plt.title("Error")
plt.savefig('./results/error.pdf')
plt.close()

# =====================================================
# LOSS CURVE
# =====================================================
loss_data = np.loadtxt(loss_log_file, delimiter=',', skiprows=1)
plt.figure()
plt.semilogy(loss_data[:,0], loss_data[:,3])
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.savefig('./results/loss_curve.pdf')
plt.close()

print("\n✅ Training complete. All logs and plots saved to ./results/")
