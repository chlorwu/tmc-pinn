# reaction-error-logging.py
import time
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS
from tqdm import tqdm
import argparse
from util import *
from model_dict import get_model

# -------------------- Seed --------------------
seed = 1
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

step_size = 1e-4
num_step = 5

# -------------------- Args --------------------
parser = argparse.ArgumentParser('Training Point Optimization')
parser.add_argument('--model', type=str, default='PINN')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
device = args.device

# -------------------- Data --------------------
res, b_left, b_right, b_upper, b_lower = get_data([0, 2 * np.pi], [0, 1], 101, 101)
res_test, _, _, _, _ = get_data([0, 2 * np.pi], [0, 1], 101, 101)

if args.model in ['PINNsFormer', 'PINNMamba']:
    res = make_time_sequence(res, num_step=10, step=step_size)
    b_left = make_time_sequence(b_left, num_step=num_step, step=step_size)
    b_right = make_time_sequence(b_right, num_step=num_step, step=step_size)
    b_upper = make_time_sequence(b_upper, num_step=num_step, step=step_size)
    b_lower = make_time_sequence(b_lower, num_step=num_step, step=step_size)

# Convert tensors to float64 initially
res = torch.tensor(res, dtype=torch.float64, requires_grad=True).to(device)
b_left = torch.tensor(b_left, dtype=torch.float64, requires_grad=True).to(device)
b_right = torch.tensor(b_right, dtype=torch.float64, requires_grad=True).to(device)
b_upper = torch.tensor(b_upper, dtype=torch.float64, requires_grad=True).to(device)
b_lower = torch.tensor(b_lower, dtype=torch.float64, requires_grad=True).to(device)

x_res, t_res = res[:, ..., 0:1], res[:, ..., 1:2]
x_left, t_left = b_left[:, ..., 0:1], b_left[:, ..., 1:2]
x_right, t_right = b_right[:, ..., 0:1], b_right[:, ..., 1:2]
x_upper, t_upper = b_upper[:, ..., 0:1], b_upper[:, ..., 1:2]
x_lower, t_lower = b_lower[:, ..., 0:1], b_lower[:, ..., 1:2]

# -------------------- Model --------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

if args.model == 'KAN':
    model = get_model(args).Model(width=[2, 5, 1], grid=5, k=3, grid_eps=1.0,
                                  noise_scale_base=0.25, device=device).to(torch.float64).to(device)
elif args.model == 'QRes':
    model = get_model(args).Model(in_dim=2, hidden_dim=256, out_dim=1, num_layer=4).to(torch.float64).to(device)
    model.apply(init_weights)
elif args.model in ['PINNsFormer', 'PINNsFormer_Enc_Only']:
    model = get_model(args).Model(in_dim=2, hidden_dim=32, out_dim=1, num_layer=1).to(torch.float64).to(device)
    model.apply(init_weights)
else:
    model = get_model(args).Model(in_dim=2, hidden_dim=1024, out_dim=1, num_layer=6).to(torch.float64).to(device)
    model.apply(init_weights)

optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe', tolerance_grad=1e-8, tolerance_change=1e-10)

print(model)
print(get_n_params(model))

# -------------------- Loss Logging --------------------
loss_track = []

if not os.path.exists('./results/'):
    os.makedirs('./results/')
log_file_path = f'./results/1dreaction_{args.model}_loss_log.txt'
with open(log_file_path, 'w') as log_file:
    log_file.write('epoch,loss_res,loss_bc,loss_ic,total_loss,precision\n')

# -------------------- Precision Switcher --------------------
class ConvergencePrecisionSwitcher:
    def __init__(self, model, patience=500):
        self.model = model
        self.patience = patience
        self.counter = 0
        self.current_precision = 'float64'

    def step(self, total_loss):
        # If loss is small enough or patience exceeded, switch to float32
        if self.counter >= self.patience and self.current_precision == 'float64':
            print("Switching model to float32 precision!")
            self.model.float()
            # Convert all input tensors globally to float32
            global x_res, t_res, x_left, t_left, x_right, t_right, x_upper, t_upper, x_lower, t_lower
            x_res = x_res.float()
            t_res = t_res.float()
            x_left = x_left.float()
            t_left = t_left.float()
            x_right = x_right.float()
            t_right = t_right.float()
            x_upper = x_upper.float()
            t_upper = t_upper.float()
            x_lower = x_lower.float()
            t_lower = t_lower.float()
            self.current_precision = 'float32'
            self.counter = 0
        else:
            self.counter += 1

switcher = ConvergencePrecisionSwitcher(model, patience=500)

# -------------------- Training Loop --------------------
num_epochs = 2000
for epoch in tqdm(range(num_epochs), desc="Training"):
    def closure():
        pred_res = model(x_res, t_res)
        pred_left = model(x_left, t_left)
        pred_right = model(x_right, t_right)
        pred_upper = model(x_upper, t_upper)
        pred_lower = model(x_lower, t_lower)

        u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=torch.ones_like(pred_res),
                                  retain_graph=True, create_graph=True)[0]
        u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res),
                                  retain_graph=True, create_graph=True)[0]

        loss_res = torch.mean((u_t - 5 * pred_res * (1 - pred_res)) ** 2)
        loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
        loss_ic = torch.mean(
            (pred_left[:, 0] - torch.exp(- (x_left[:, 0] - torch.pi) ** 2 / (2 * (torch.pi / 4) ** 2))) ** 2)

        total_loss = loss_res + loss_bc + loss_ic
        optim.zero_grad()
        total_loss.backward()

        # Step the precision switcher
        switcher.step(total_loss)

        # Append loss + current precision
        loss_track.append([loss_res.item(), loss_bc.item(), loss_ic.item(), total_loss.item(), switcher.current_precision])
        return total_loss

    optim.step(closure)

    # Print losses every 10 epochs
    if epoch % 10 == 0 or switcher.counter == 0:
        print(f"Epoch {epoch+1}: Loss Res={loss_track[-1][0]:.3e}, BC={loss_track[-1][1]:.3e}, IC={loss_track[-1][2]:.3e}, Total={loss_track[-1][3]:.3e}, Precision={loss_track[-1][4]}")

    # Log losses to file
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"{epoch+1},{loss_track[-1][0]:.8e},{loss_track[-1][1]:.8e},{loss_track[-1][2]:.8e},{loss_track[-1][3]:.8e},{loss_track[-1][4]}\n")

# -------------------- Final Loss Summary --------------------
print('Final Loss Res: {:4f}, BC: {:4f}, IC: {:4f}'.format(loss_track[-1][0], loss_track[-1][1], loss_track[-1][2]))
print('Final Train Loss: {:4f}, Precision={}'.format(loss_track[-1][3], loss_track[-1][4]))

torch.save(model.state_dict(), f'./results/1dreaction_{args.model}_point.pt')

# -------------------- Prediction & Visualization --------------------
if args.model in ['PINNsFormer', 'PINNMamba']:
    res_test = make_time_sequence(res_test, num_step=5, step=1e-4)
res_test = torch.tensor(res_test, dtype=torch.float64, requires_grad=True).to(device)
x_test, t_test = res_test[:, ..., 0:1], res_test[:, ..., 1:2]

with torch.no_grad():
    pred = model(x_test, t_test)[:, 0:1]
    pred = pred.cpu().detach().numpy()
pred = pred.reshape(101, 101)

def h(x):
    return np.exp(- (x - np.pi) ** 2 / (2 * (np.pi / 4) ** 2))
def u_ana(x, t):
    return h(x) * np.exp(5 * t) / (h(x) * np.exp(5 * t) + 1 - h(x))

res_test, _, _, _, _ = get_data([0, 2 * np.pi], [0, 1], 101, 101)
u = u_ana(res_test[:, 0], res_test[:, 1]).reshape(101, 101)

rl1 = np.sum(np.abs(u - pred)) / np.sum(np.abs(u))
rl2 = np.sqrt(np.sum((u - pred) ** 2) / np.sum(u ** 2))
print('relative L1 error: {:4f}'.format(rl1))
print('relative L2 error: {:4f}'.format(rl2))

# -------------------- Plots --------------------
plt.figure(figsize=(4, 3))
plt.imshow(pred, extent=[0,1,1,0], aspect='auto')
plt.xlabel('x'); plt.ylabel('t'); plt.title('Predicted u(x,t)'); plt.colorbar(); plt.tight_layout()
plt.savefig(f'./results/1d_reaction_{args.model}_{num_step}_{step_size}_pred.pdf', bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.imshow(u, extent=[0,1,1,0], aspect='auto')
plt.xlabel('x'); plt.ylabel('t'); plt.title('Exact u(x,t)'); plt.colorbar(); plt.tight_layout()
plt.savefig('./results/1d_reaction_exact.pdf', bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.imshow(pred - u, extent=[0,1,1,0], aspect='auto', cmap='coolwarm', vmin=-0.15, vmax=0.15)
plt.xlabel('x'); plt.ylabel('t'); plt.title('Absolute Error'); plt.colorbar(); plt.tight_layout()
plt.savefig(f'./results/1d_reaction_{args.model}_{num_step}_{step_size}_error.pdf', bbox_inches='tight')

# -------------------- Loss Curve --------------------
try:
    loss_data = np.loadtxt(log_file_path, delimiter=',', skiprows=1)
    epochs = loss_data[:, 0]
    loss_res = loss_data[:, 1]
    loss_bc = loss_data[:, 2]
    loss_ic = loss_data[:, 3]
    total_loss = loss_data[:, 4]

    plt.figure(figsize=(10, 6))
    plt.semilogy(epochs, total_loss, 'b-', label='Total Loss', linewidth=2)
    plt.semilogy(epochs, loss_res, 'r--', label='Residual Loss', linewidth=1.5, alpha=0.7)
    plt.semilogy(epochs, loss_bc, 'g--', label='BC Loss', linewidth=1.5, alpha=0.7)
    plt.semilogy(epochs, loss_ic, 'm--', label='IC Loss', linewidth=1.5, alpha=0.7)
    plt.xlabel('Epoch'); plt.ylabel('Loss (log scale)'); plt.title(f'Training Loss vs Epoch - {args.model}')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f'./results/1d_reaction_{args.model}_loss_curve.pdf', bbox_inches='tight')
    print(f'Loss curve saved to: ./results/1d_reaction_{args.model}_loss_curve.pdf')
except Exception as e:
    print(f'Warning: Could not plot loss curves: {e}')
