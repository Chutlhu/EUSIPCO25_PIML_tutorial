import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

class PINN_Wave1D(nn.Module):
    def __init__(self, 
        model_arch, 
        wave_speed=1.0, wave_mode=1, domain_length=1.0, 
        samples_sizes = {
            'N_pde': 2048,
            'N_ic' : 1024,
            'N_bc' : 256
        },
        loss_weights = {
            'pde' : 1., 
            'ic' : 1., 
            'bc' : 1.
        },
        device='cpu', 
    ):
        """
        Initialize PINN for wave equation
        
        Args:
            model_arch: PyTorch model class or instance
            wave_speed: Wave propagation speed (c)
            wave_mode: Wave mode number (k = mode * pi)
            domain_length: Length of spatial domain
            device: Computing device
            N_c, N_i, N_b: Number of collocation, initial, boundary points
        """
        super().__init__()
        
        self.device = device
        self.c = wave_speed
        self.k = 2 * wave_mode * np.pi / domain_length
        self.omega = self.c * self.k
        self.L = domain_length
        print(f"Wave PINN for c={self.c}, mode={wave_mode}, k={self.k:.3f}, L={self.L:.2f}")
        
        # Sampling parameters
        self.N_pde = samples_sizes['N_pde']
        self.N_ic = samples_sizes['N_ic'] if 'N_ic' in samples_sizes else 0
        self.N_bc = samples_sizes['N_bc'] if 'N_bc' in samples_sizes else 0

        # Initialize model
        self.model = model_arch.to(device)
            
        # Loss weights
        self.w_pde = loss_weights['pde']
        self.w_ic = loss_weights['ic'] if 'ic' in loss_weights else 0.
        self.w_bc = loss_weights['bc'] if 'bc' in loss_weights else 0.
        
        sum_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model params: {sum_params/1e3:.1f}k")

    def sample_collocation(self, Nc):
        """Sample collocation points in domain [0,1] x [0,L]"""
        t = torch.rand(Nc, 1)
        x = torch.rand(Nc, 1) * self.L
        return torch.cat([t, x], dim=1).to(self.device)

    def sample_ic(self, Ni):
        """Sample initial condition points"""
        x = torch.rand(Ni, 1) * self.L
        t = torch.zeros_like(x)
        u0 = torch.sin(self.k * x)
        ut0 = -self.omega * torch.cos(self.k * x)
        return torch.cat([t, x], 1).to(self.device), u0.to(self.device), ut0.to(self.device)

    def sample_bc(self, Nb):
        """Sample boundary condition points for periodic BC"""
        t = torch.rand(Nb, 1)
        x0 = torch.zeros_like(t)
        xL = torch.full_like(t, self.L)
        return (torch.cat([t, x0], 1).to(self.device),
                torch.cat([t, xL], 1).to(self.device))

    def derivatives(self, u, tx):
        """Compute derivatives using autograd"""
        grads = torch.autograd.grad(u, tx, torch.ones_like(u), create_graph=True)[0]
        ut, ux = grads[:, :1], grads[:, 1:]
        utt = torch.autograd.grad(ut, tx, torch.ones_like(ut), create_graph=True)[0][:, :1]
        uxx = torch.autograd.grad(ux, tx, torch.ones_like(ux), create_graph=True)[0][:, 1:]
        return ut, ux, utt, uxx

    def analytical_solution(self, t, x):
        """Compute analytical solution"""
        return torch.sin(self.k * x - self.omega * t)

    def train(self, steps_adam=1500, lr=1e-3, log_every=100, plot_live=False):
        """Train the PINN model"""
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        hist = []
        
        # Evaluation points for live plotting
        if plot_live:
            Nx = 400
            x_eval = torch.linspace(0, self.L, Nx)[:, None]
            t_eval = 0.35
            tx_eval = torch.cat([torch.full_like(x_eval, t_eval), x_eval], dim=1).to(self.device)

        for step in range(steps_adam):
            opt.zero_grad(set_to_none=True)

            # (1) PDE loss
            tx_c = self.sample_collocation(self.N_pde).requires_grad_(True)
            u_c = self.model(tx_c)
            ut_c, ux_c, utt_c, uxx_c = self.derivatives(u_c, tx_c)
            r = utt_c - self.c**2 * uxx_c
            L_pde = torch.mean(r**2)

            # (2) IC loss
            tx_i, u0, ut0 = self.sample_ic(self.N_ic)
            tx_i = tx_i.requires_grad_(True)
            u_i = self.model(tx_i)
            ut_i, _, _, _ = self.derivatives(u_i, tx_i)
            L_ic = torch.mean((u_i - u0)**2) + torch.mean((ut_i - ut0)**2)

            # (3) Periodic BC loss
            tx0, txL = self.sample_bc(self.N_bc)
            tx0 = tx0.requires_grad_(True)
            txL = txL.requires_grad_(True)
            u0b = self.model(tx0)
            uLb = self.model(txL)
            _, ux0, _, _ = self.derivatives(u0b, tx0)
            _, uxL, _, _ = self.derivatives(uLb, txL)
            L_bc = torch.mean((u0b - uLb)**2) + torch.mean((ux0 - uxL)**2)

            # (4) Total loss & step
            loss = self.w_pde * L_pde + self.w_ic * L_ic + self.w_bc * L_bc
            loss.backward()
            opt.step()

            if step % log_every == 0:
                parts = {
                    'pde': float(L_pde.detach().cpu()),
                    'ic': float(L_ic.detach().cpu()),
                    'bc': float(L_bc.detach().cpu())
                }
                hist.append((step, float(loss.detach().cpu()), parts))
                print(f"[Adam {step:04d}] total={hist[-1][1]:.3e} parts={parts}")

                if plot_live:
                    self._plot_live_training(hist, tx_eval, x_eval, t_eval, step=step, total_steps=steps_adam)

        return hist

    def _plot_live_training(self, hist, tx_eval, x_eval, t_eval, step=0, total_steps=0, title_suffix=''):
        """Live plotting during training"""
        
        try:
            from IPython.display import clear_output
            clear_output(wait=True)
        except ImportError:
            pass  # Skip clearing output if IPython not available
        
        steps, totals, comps = zip(*hist)
        pde = [c['pde'] for c in comps]
        ic = [c['ic'] for c in comps] if 'ic' in comps[0] else None
        bc = [c['bc'] for c in comps] if 'bc' in comps[0] else None
        data = [c['data'] for c in comps] if 'data' in comps[0] else None

        with torch.no_grad():
            u_pred = self.model(tx_eval).cpu()
        u_ref = self.analytical_solution(t_eval, x_eval.to(self.device)).cpu()
        
        fig, axarr = plt.subplots(1, 3, figsize=(12, 4))
        axarr[0].plot(x_eval.squeeze(), u_ref.squeeze(), label='Analytical True', lw=2)
        axarr[0].plot(x_eval.squeeze(), u_pred.squeeze(), '--', label='PINN', lw=2)
        axarr[0].legend()
        axarr[0].set_title(f"Time slice t={t_eval:.2f}")
        axarr[0].set_xlabel("x")
        axarr[0].set_ylabel("u")

        axarr[1].plot(steps, totals, label='Total Loss')
        axarr[1].set_xlim([0, total_steps])
        axarr[1].set_title("Total Loss")
        axarr[1].set_xlabel("Steps")
        axarr[1].grid(True)
        
        axarr[2].plot(steps, pde, label='PDE')
        if ic is not None:
            axarr[2].plot(steps, ic, label='IC')
        if bc is not None:
            axarr[2].plot(steps, bc, label='BC')
        if data is not None:
            axarr[2].plot(steps, data, label='Data')
        axarr[2].set_xlim([0, total_steps])
        axarr[2].set_title("Loss Components")
        axarr[2].set_xlabel("Steps")
        axarr[2].legend()
        axarr[2].grid(True)
        
        plt.suptitle(f'Training loop {step}/{total_steps}' + title_suffix)

        plt.tight_layout()
        plt.show()
                

    def __call__(self, tx):
        return self.model(tx)

    def predict(self, t, x):
        """Make predictions at given time and space points"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(t, (int, float)):
                t = torch.full_like(x, t)
            tx = torch.cat([t.to(self.device), x.to(self.device)], dim=1)
            return self.model(tx)


class PINN_Wave1D_inverse(PINN_Wave1D):
    def __init__(self, 
        model_arch, 
        wave_speed=1.0, wave_mode=1, domain_length=1.0, 
        samples_sizes = {
            'N_pde': 2048,
        },
        loss_weights = {
            'pde' : 1., 
            'data' : 1.,
        },
        device='cpu',
    ):
        super().__init__(model_arch, wave_speed, wave_mode, domain_length, samples_sizes, loss_weights, device)
        # Make wave speed a learnable parameter
        # self.c = torch.nn.Parameter(torch.tensor(wave_speed, dtype=torch.float32, device=device))
        # print("Inverse problem: Wave speed 'c' is now a learnable parameter.")
        self.w_pde = loss_weights['pde']
        self.w_data = loss_weights['data']
        
        self.c_hat = torch.tensor(0.1, device=device, dtype=torch.float32)
        self.c_hat = torch.nn.Parameter(self.c_hat, requires_grad=True)
        self.learning_c = True
        if self.learning_c:
            print("Inverse problem: Wave speed 'c_hat' is now a learnable parameter.")
        else:
            self.c_hat = self.c

    def train(self, tx_data, u_data, steps_adam=1500, lr=1e-3, log_every=100, plot_live=False):
        """Train the PINN model"""
        self.model.train()
        if self.learning_c:
            print(f"Learning wave speed c_hat, initial value: {self.c_hat.item():.3f}")
            opt = torch.optim.Adam(list(self.model.parameters()) + [self.c_hat], lr=lr)
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        hist = []
        title_suffix = ''
        
        # Evaluation points for live plotting
        if plot_live:
            x_eval = torch.linspace(0, self.L, 400)[:, None]
            t_eval = 0.35
            tx_eval = torch.cat([torch.full_like(x_eval, t_eval), x_eval], dim=1).to(self.device)

        for step in range(steps_adam):
            opt.zero_grad(set_to_none=True)

            # (1) PDE loss
            tx_c = self.sample_collocation(self.N_pde).requires_grad_(True)
            u_c = self.model(tx_c)
            ut_c, ux_c, utt_c, uxx_c = self.derivatives(u_c, tx_c)
            r = utt_c - self.c_hat**2 * uxx_c
            L_pde = torch.mean(r**2)

            # (4) Data / Supervised losses
            u_d = self.model(tx_data)
            L_data = torch.mean((u_d - u_data)**2)

            # (6) total loss & step
            loss = self.w_pde*L_pde + self.w_data*L_data
            loss.backward()
            opt.step()

            if step % log_every == 0:
                parts = {
                    'pde': float(L_pde.detach().cpu()),
                    'data': float(L_data.detach().cpu()),
                }
                hist.append((step, float(loss.detach().cpu()), parts))
                print(f"[Adam {step:04d}] total={hist[-1][1]:.3e} parts={parts}")
                
                if self.learning_c:
                    current_val_c = self.c_hat.detach().cpu()
                    target_c = self.c
                    title_suffix = f" - C_est = {current_val_c:.3f} --> C_true = {target_c:.3f}"

                if plot_live:
                    self._plot_live_training(hist, tx_eval, x_eval, t_eval, step=step, total_steps=steps_adam, title_suffix=title_suffix)

        if plot_live:
            self._plot_live_training(hist, tx_eval, x_eval, t_eval, step=step, total_steps=steps_adam, title_suffix=title_suffix)

        return hist


# Plain MLP architecture
class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=128, depth=4):
        super().__init__()
        layers = []
        for i in range(depth):
            layers += [nn.Linear(in_dim if i == 0 else hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
    
    def forward(self, tx):
        return self.net(tx)

# Example usage:
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and train PINN
    pinn = WavePINN(
        model_arch=MLP,
        wave_speed=1.0,
        wave_mode=1,
        domain_length=1.0,
        device=device
    )
    
    start = time.time()
    hist = pinn.train(steps_adam=2000, lr=1e-3, log_every=100, plot_live=True)
    print(f"Training completed in {time.time()-start:.2f} seconds")