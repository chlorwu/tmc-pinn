import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS,Adam
from tqdm import tqdm
import os
import argparse
from util import *
from model_dict import get_model

#torch.set_float64_matmul_precision('high')
#torch.backends.cuda.matmul.allow_tf32 = False

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
step_size = 1e-4
num_step=5
beta=50

parser = argparse.ArgumentParser('Training Point Optimization')
parser.add_argument('--model', type=str, default='PINN')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
device = args.device

res, b_left, b_right, b_upper, b_lower = get_data([0, 2 * np.pi], [0, 1], 401, 401)
res_test, b_left_test, _, _, _ = get_data([0, 2 * np.pi], [0, 1], 101, 101)

if args.model == 'PINNsFormer' or args.model == 'PINNMamba':
    res = make_time_sequence(res, num_step=num_step, step=step_size)
    b_left = make_time_sequence(b_left, num_step=num_step, step=step_size)
    b_right = make_time_sequence(b_right, num_step=num_step, step=step_size)
    b_upper = make_time_sequence(b_upper, num_step=num_step, step=step_size)
    b_lower = make_time_sequence(b_lower, num_step=num_step, step=step_size)
    
if args.model == 'PINNsFormer' or args.model == 'PINNMamba':
    res_test = make_time_sequence(res_test, num_step=num_step, step=step_size)

res = torch.tensor(res, dtype=torch.bfloat16, requires_grad=True).to(device)
b_left = torch.tensor(b_left, dtype=torch.bfloat16, requires_grad=True).to(device)
b_right = torch.tensor(b_right, dtype=torch.bfloat16, requires_grad=True).to(device)
b_upper = torch.tensor(b_upper, dtype=torch.bfloat16, requires_grad=True).to(device)
b_lower = torch.tensor(b_lower, dtype=torch.bfloat16, requires_grad=True).to(device)

x_res, t_res = res[:, ..., 0:1], res[:, ..., 1:2]
x_left, t_left = b_left[:, ..., 0:1], b_left[:, ..., 1:2]
x_right, t_right = b_right[:, ..., 0:1], b_right[:, ..., 1:2]
x_upper, t_upper = b_upper[:, ..., 0:1], b_upper[:, ..., 1:2]
x_lower, t_lower = b_lower[:, ..., 0:1], b_lower[:, ..., 1:2]

print(t_res)
def init_weights(m):
    if isinstance(m, nn.Linear):
        #if(m.bias):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)


if args.model == 'KAN':
    model = get_model(args).Model(width=[2, 5, 5, 1], grid=5, k=3, grid_eps=1.0, \
                                  noise_scale_base=0.25, device=device).to(torch.bfloat16).to(device)
elif args.model == 'QRes':
    model = get_model(args).Model(in_dim=2, hidden_dim=256, out_dim=1, num_layer=4).to(torch.bfloat16).to(device)
    model.apply(init_weights)
elif args.model == 'PINNsFormer' or args.model == 'PINNsFormer_Enc_Only':
    model = get_model(args).Model(in_dim=2, hidden_dim=32, out_dim=1, num_layer=1).to(torch.bfloat16).to(device)
    model.apply(init_weights)
else:
    model = get_model(args).Model(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4).to(torch.bfloat16).to(device)
    model.apply(init_weights)



#optim = Adam(model.parameters(),lr=1e-6)
optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe', tolerance_grad = 1e-8, tolerance_change= 1e-10)
print(model)
print(get_n_params(model))

loss_track = []

gradient_stats = []

print(model.named_parameters())

# Set up loss log file
if not os.path.exists('./results/'):
    os.makedirs('./results/')
loss_log_file = f'./results/1dconvection_{args.model}_{num_step}_{step_size}_{beta}_loss_log.txt'
with open(loss_log_file, 'w') as f:
    f.write('epoch,loss_res,loss_bc,loss_ic,total_loss\n')

# Open log file for writing FLOPs/FLOPS
flops_log_file_path = f'./results/1dconvection_{args.model}_{num_step}_{step_size}_{beta}_flops_log.txt'
with open(flops_log_file_path, 'w') as flops_log_file:
    flops_log_file.write('epoch,forward_flops,backward_flops,total_flops,forward_time,backward_time,total_time,flops_per_sec\n')

# Function to estimate FLOPs for a forward pass
def estimate_flops(model, input_shape):
    """
    Estimate FLOPs for forward pass.
    For Linear layers: FLOPs = 2 * in_features * out_features * batch_size
    """
    flops = 0
    batch_size = input_shape[0]
    
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Matrix multiplication: in_features * out_features * batch_size
            # Plus bias addition: out_features * batch_size
            flops += (2 * module.in_features * module.out_features + module.out_features) * batch_size
        # Add FLOPs for activation functions (approximate)
        elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
            flops += batch_size * module.out_features if hasattr(module, 'out_features') else batch_size
    
    return flops

# Estimate FLOPs for one forward pass (using x_res shape as reference)
# Handle different input shapes: (N, 1) or (N, L, 1)
if len(x_res.shape) >= 2:
    sample_batch_size = x_res.shape[0]
    # If 3D, multiply by sequence length for total operations
    if len(x_res.shape) == 3:
        sample_batch_size = x_res.shape[0] * x_res.shape[1]
else:
    sample_batch_size = 1
forward_flops_per_pass = estimate_flops(model, (sample_batch_size,))
# Backward pass typically requires ~2x FLOPs of forward pass
backward_flops_per_pass = forward_flops_per_pass * 2

flops_track = []

for i in tqdm(range(50000)): 
    # Use a list to store timing info (mutable, can be modified in closure)
    timing_info = [0.0, 0.0]  # [forward_time, backward_time]
    
    # Measure time for the entire closure (forward + backward)
    epoch_start_time = time.time()
    
    def closure():
        # Measure forward pass time
        forward_start_time = time.time()
        
        pred_res = model(x_res, t_res)
        pred_left = model(x_left, t_left)
        pred_right = model(x_right, t_right)
        pred_upper = model(x_upper, t_upper)
        pred_lower = model(x_lower, t_lower)

        u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                  create_graph=True)[0]
        u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                  create_graph=True)[0]

        loss_res = torch.mean((u_t + beta * u_x) ** 2)
        loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
        loss_ic = torch.mean((pred_left[:,0] - torch.sin(x_left[:,0])) ** 2)

        forward_end_time = time.time()
        timing_info[0] = forward_end_time - forward_start_time
        
        loss_track.append([loss_res.item(), loss_bc.item(), loss_ic.item()])
        #print('Loss Res: {:4f}, Loss_BC: {:4f}, Loss_IC: {:4f}'.format(loss_track[-1][0], loss_track[-1][1], loss_track[-1][2]))
        loss = loss_res + loss_bc + loss_ic
        #loss = 1000*loss
        #loss = loss_ic
        optim.zero_grad()
        
        # Measure backward pass time
        backward_start_time = time.time()
        loss.backward()
        backward_end_time = time.time()
        timing_info[1] = backward_end_time - backward_start_time
        
        return loss


    optim.step(closure)
    
    epoch_end_time = time.time()
    total_time = epoch_end_time - epoch_start_time
    
    # Calculate FLOPs and FLOPS
    # Count number of forward passes (5 model calls: pred_res, pred_left, pred_right, pred_upper, pred_lower)
    # Plus 2 gradient computations
    num_forward_passes = 5
    num_grad_computations = 2
    
    # Estimate FLOPs: each forward pass through model + gradient computations
    # Gradient computations add significant FLOPs (roughly 2-3x forward pass)
    forward_flops = forward_flops_per_pass * num_forward_passes
    # Gradient computations are computationally expensive
    grad_flops = forward_flops_per_pass * num_grad_computations * 2  # 2x multiplier for gradient computation
    total_forward_flops = forward_flops + grad_flops
    total_backward_flops = backward_flops_per_pass * num_forward_passes  # Approximate backward FLOPs
    total_flops = total_forward_flops + total_backward_flops
    
    # Get timing from closure
    forward_time_measured = timing_info[0] if timing_info[0] > 0 else total_time * 0.6
    backward_time_measured = timing_info[1] if timing_info[1] > 0 else total_time * 0.4
    
    # Calculate FLOPS (FLOPs per second)
    if total_time > 0:
        flops_per_sec = total_flops / total_time
    else:
        flops_per_sec = 0.0
    
    flops_track.append([total_forward_flops, total_backward_flops, total_flops, forward_time_measured, backward_time_measured, total_time, flops_per_sec])

    # Log loss to file
    if len(loss_track) > 0:
        loss_res_val, loss_bc_val, loss_ic_val = loss_track[-1]
        total_loss_val = loss_res_val + loss_bc_val + loss_ic_val
        with open(loss_log_file, 'a') as f:
            f.write(f'{i},{loss_res_val:.10e},{loss_bc_val:.10e},{loss_ic_val:.10e},{total_loss_val:.10e}\n')
    
    # Log FLOPs/FLOPS to file after each epoch
    with open(flops_log_file_path, 'a') as flops_log_file:
        flops_log_file.write(f'{i+1},{total_forward_flops:.2e},{total_backward_flops:.2e},{total_flops:.2e},'
                            f'{forward_time_measured:.6f},{backward_time_measured:.6f},{total_time:.6f},{flops_per_sec:.2e}\n')

    grad_norms = []
    grad_means = []
    grad_stds = []


    if i>50:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad).item()  
                    grad_mean = param.grad.mean().item()       
                    grad_std = param.grad.std().item()      
                    grad_norms.append(grad_norm)
                    grad_means.append(grad_mean)
                    grad_stds.append(grad_std)

           
            gradient_stats.append({
                'step': i,
                'grad_norms': grad_norms,
                'grad_means': grad_means,
                'grad_stds': grad_stds
            })



print('Loss Res: {:4f}, Loss_BC: {:4f}, Loss_IC: {:4f}'.format(loss_track[-1][0], loss_track[-1][1], loss_track[-1][2]))
print('Train Loss: {:4f}'.format(np.sum(loss_track[-1])))

if not os.path.exists('./results/'):
    os.makedirs('./results/')

torch.save(model.state_dict(), f'./results/1dconvection_{args.model}_{num_step}_{step_size}.pt')

# Visualize


res_test = torch.tensor(res_test, dtype=torch.bfloat16, requires_grad=True).to(device)
x_test, t_test = res_test[:, ..., 0:1], res_test[:, ..., 1:2]
x_left_test, t_left_test = b_left_test[:, ..., 0:1], b_left_test[:, ..., 1:2]


#print(t_test)
#with torch.no_grad():
pred = model(x_test, t_test)[:, 0:1]
u_x = torch.autograd.grad(pred, x_test, grad_outputs=torch.ones_like(pred), retain_graph=True,
                                  create_graph=True)[0]
u_t = torch.autograd.grad(pred, t_test, grad_outputs=torch.ones_like(pred), retain_graph=True,
                                  create_graph=True)[0]
#loss_res = (u_t + 50 * u_x) ** 2


    #print(pred.shape)
    #pred = model(x_test, t_test)[:, 0:1]
#loss_res = loss_res.cpu().detach().numpy()
#loss_res = loss_res.reshape(101, 101)

#print(loss_res)

pred = pred.float().cpu().numpy()

#print(pred.shape) 
pred = pred.reshape(101, 101)


def u_res(x, t):
    #print(x.shape)
    #print(t.shape)
    return np.sin(x - beta * t)


res_test, _, _, _, _ = get_data([0, 2 * np.pi], [0, 1], 101, 101)
u = u_res(res_test[:, 0], res_test[:, 1]).reshape(101, 101)

rl1 = np.sum(np.abs(u - pred)) / np.sum(np.abs(u))
rl2 = np.sqrt(np.sum((u - pred) ** 2) / np.sum(u ** 2))

print(beta)
print('relative L1 error: {:4f}'.format(rl1))
print('relative L2 error: {:4f}'.format(rl2))

plt.figure(figsize=(4, 3))
plt.imshow(pred,extent=[0,np.pi*2,1,0], aspect='auto')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Predicted u(x,t)')
plt.colorbar()
plt.tight_layout()
#plt.axis('off')
plt.savefig(f'./results/convection_{args.model}_{num_step}_{step_size}_{beta}_pred.pdf', bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.imshow(u,extent=[0,np.pi*2,1,0], aspect='auto')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Exact u(x,t)')
plt.colorbar()
plt.tight_layout()
#plt.axis('off')
plt.savefig('./results/convection_exact_{beta}.pdf', bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.imshow(pred - u, extent=[0,np.pi*2,1,0], aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
plt.xlabel('x')
plt.ylabel('t')
plt.title('Absolute Error')
plt.colorbar()
plt.tight_layout()
#plt.axis('off')
plt.savefig(f'./results/convection_{args.model}_{num_step}_{step_size}_{beta}_error.pdf', bbox_inches='tight')

grad_norms_history = [stats['grad_norms'] for stats in gradient_stats]
grad_means_history = [stats['grad_means'] for stats in gradient_stats]
grad_stds_history = [stats['grad_stds'] for stats in gradient_stats]


plt.figure(figsize=(10, 6))
for i in range(len(grad_norms_history[0])):
    plt.plot([stats[i] for stats in grad_norms_history], label=f'Layer {i+1}')
plt.xlabel('Training Step')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norms During Training')
plt.legend()
plt.grid()
plt.savefig(f'./results/1d_convection_{args.model}_{beta}_fp64_gradient_norms.pdf')  
plt.close()  


plt.figure(figsize=(10, 6))
for i in range(len(grad_means_history[0])):
    plt.plot([stats[i] for stats in grad_means_history], label=f'Layer {i+1}')
plt.xlabel('Training Step')
plt.ylabel('Gradient Mean')
plt.title('Gradient Means During Training')
plt.legend()
plt.grid()
plt.savefig(f'./results/1d_convection_{args.model}_{beta}_fp64_gradient_means.pdf')  
plt.close()  


plt.figure(figsize=(10, 6))
for i in range(len(grad_stds_history[0])):
    plt.plot([stats[i] for stats in grad_stds_history], label=f'Layer {i+1}')
plt.xlabel('Training Step')
plt.ylabel('Gradient Std')
plt.title('Gradient Stds During Training')
plt.legend()
plt.grid()
plt.savefig(f'./results/1d_convection_{args.model}_{beta}_fp64_gradient_stds.pdf')  
plt.close()  

# Read loss log file and create loss vs epoch plot
try:
    loss_data = np.loadtxt(loss_log_file, delimiter=',', skiprows=1)
    epochs = loss_data[:, 0]
    loss_res = loss_data[:, 1]
    loss_bc = loss_data[:, 2]
    loss_ic = loss_data[:, 3]
    total_loss = loss_data[:, 4]
    
    plt.figure(figsize=(12, 8))
    
    # Plot individual loss components
    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss_res, label='Loss Res', alpha=0.7)
    plt.plot(epochs, loss_bc, label='Loss BC', alpha=0.7)
    plt.plot(epochs, loss_ic, label='Loss IC', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Individual Loss Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use log scale for better visualization
    
    # Plot total loss
    plt.subplot(2, 1, 2)
    plt.plot(epochs, total_loss, label='Total Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Total Loss vs Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use log scale for better visualization
    
    plt.tight_layout()
    plt.savefig(f'./results/1d_convection_{args.model}_{beta}_fp64_loss_vs_epoch.pdf', bbox_inches='tight')
    plt.close()
    
    print(f'\nLoss log file saved at: {loss_log_file}')
    print(f'Loss plot saved at: ./results/1d_convection_{args.model}_{beta}_fp64_loss_vs_epoch.pdf')
    
except Exception as e:
    print(f'Warning: Could not read loss log file or create plot: {e}')

# Plot FLOPs/FLOPS curves from log file
try:
    # Load the FLOPs data
    flops_data = np.loadtxt(flops_log_file_path, delimiter=',', skiprows=1)
    epochs_flops = flops_data[:, 0]
    forward_flops = flops_data[:, 1]
    backward_flops = flops_data[:, 2]
    total_flops = flops_data[:, 3]
    forward_time = flops_data[:, 4]
    backward_time = flops_data[:, 5]
    total_time = flops_data[:, 6]
    flops_per_sec = flops_data[:, 7]
    
    # Create FLOPs vs epoch plot
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: FLOPs breakdown
    plt.subplot(2, 2, 1)
    plt.semilogy(epochs_flops, total_flops, 'b-', label='Total FLOPs', linewidth=2)
    plt.semilogy(epochs_flops, forward_flops, 'r--', label='Forward FLOPs', linewidth=1.5, alpha=0.7)
    plt.semilogy(epochs_flops, backward_flops, 'g--', label='Backward FLOPs', linewidth=1.5, alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('FLOPs (log scale)', fontsize=12)
    plt.title(f'FLOPs vs Epoch - {args.model}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: FLOPS (FLOPs per second)
    plt.subplot(2, 2, 2)
    plt.plot(epochs_flops, flops_per_sec, 'purple', label='FLOPS', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('FLOPS (FLOPs per second)', fontsize=12)
    plt.title(f'FLOPS vs Epoch - {args.model}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Subplot 3: Time breakdown
    plt.subplot(2, 2, 3)
    plt.plot(epochs_flops, total_time, 'b-', label='Total Time', linewidth=2)
    plt.plot(epochs_flops, forward_time, 'r--', label='Forward Time', linewidth=1.5, alpha=0.7)
    plt.plot(epochs_flops, backward_time, 'g--', label='Backward Time', linewidth=1.5, alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title(f'Time per Epoch - {args.model}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Cumulative FLOPs
    plt.subplot(2, 2, 4)
    cumulative_flops = np.cumsum(total_flops)
    plt.semilogy(epochs_flops, cumulative_flops, 'orange', label='Cumulative FLOPs', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cumulative FLOPs (log scale)', fontsize=12)
    plt.title(f'Cumulative FLOPs - {args.model}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'./results/1d_convection_{args.model}_{num_step}_{step_size}_{beta}_flops_curve.pdf', bbox_inches='tight')
    plt.close()
    print(f'FLOPs/FLOPS curve saved to: ./results/1d_convection_{args.model}_{num_step}_{step_size}_{beta}_flops_curve.pdf')
    print(f'FLOPs log file location: {os.path.abspath(flops_log_file_path)}')
except Exception as e:
    print(f'Warning: Could not plot FLOPs/FLOPS curves: {e}')