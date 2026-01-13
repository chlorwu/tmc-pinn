import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

# =====================
# CONFIG
# =====================
TOTAL_EPOCHS = 50000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

LOG_PATH = os.path.join(RESULTS_DIR, "adam32_convection_log.csv")

# =====================
# MODEL
# =====================
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

model = PINN().to(DEVICE).float()

# =====================
# DATA (x, t)
# =====================
N_COL = 2000
x = torch.rand(N_COL, 1, device=DEVICE)
t = torch.rand(N_COL, 1, device=DEVICE)

XT = torch.cat([x, t], dim=1)
XT.requires_grad_(True)

# =====================
# CONVECTION PDE
# u_t + c u_x = 0
# =====================
c = 1.0

def compute_losses():
    u = model(XT)

    grads = torch.autograd.grad(
        u, XT,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]

    loss_res = torch.mean((u_t + c * u_x) ** 2)

    # Initial condition: u(x, 0) = sin(pi x)
    mask_ic = (t < 0.01).squeeze()
    u_ic = model(XT[mask_ic])
    x_ic = x[mask_ic]
    loss_ic = torch.mean((u_ic - torch.sin(torch.pi * x_ic)) ** 2)

    loss_total = loss_res + loss_ic
    return loss_total, loss_res, loss_ic

# =====================
# OPTIMIZER (Adam FP32 ONLY)
# =====================
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# =====================
# LOGGING
# =====================
logfile = open(LOG_PATH, "w")
logfile.write("epoch,loss_total,loss_res,loss_ic,precision,optimizer,step_time\n")
logfile.flush()

# =====================
# TRAINING LOOP
# =====================
print("🚀 Training convection PINN with Adam FP32 only\n")

for epoch in range(1, TOTAL_EPOCHS + 1):

    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)
    loss_total, loss_res, loss_ic = compute_losses()
    loss_total.backward()
    optimizer.step()

    step_time = time.time() - t0

    log_line = (
        f"{epoch},"
        f"{loss_total.item()},"
        f"{loss_res.item()},"
        f"{loss_ic.item()},"
        f"fp32,"
        f"Adam,"
        f"{step_time}"
    )

    logfile.write(log_line + "\n")
    logfile.flush()
    os.fsync(logfile.fileno())

    if epoch % 1000 == 0:
        print(
            f"[{epoch}/{TOTAL_EPOCHS}] "
            f"Loss={loss_total.item():.3e} "
            f"(res={loss_res.item():.3e}, ic={loss_ic.item():.3e})"
        )

logfile.close()
print("\n✅ Training complete (Adam FP32 only)")
