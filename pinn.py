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
        
        self.pred_hist = []

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

    def train(self, 
            steps_adam=1500, lr=1e-3, 
            steps_lbfgs = 0,
            log_every=100, plot_live=False):
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
                    self._plot_live_training(hist, tx_eval, x_eval, t_eval, step=step, total_steps=steps_adam + steps_lbfgs, title_suffix = ' - Adam steps')

        self._plot_live_training(hist, tx_eval, x_eval, t_eval, step=step, total_steps=steps_adam + steps_lbfgs, title_suffix = ' - Adam steps')

        # L-BFGS refinement
        if steps_lbfgs > 0:
            print(f"Starting L-BFGS refinement for {steps_lbfgs} steps...")
            opt_lbfgs = torch.optim.LBFGS(
                self.model.parameters(), 
                lr=1.0,  # L-BFGS typically uses lr=1.0
                line_search_fn='strong_wolfe'
            )
            
            def closure():
                opt_lbfgs.zero_grad()

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

                # (4) Total loss
                loss = self.w_pde * L_pde + self.w_ic * L_ic + self.w_bc * L_bc
                loss.backward()
                
                # Store components for logging (closure can be called multiple times)
                if not hasattr(closure, 'loss_components'):
                    closure.loss_components = {}
                closure.loss_components.update({
                    'pde': float(L_pde.detach().cpu()),
                    'ic': float(L_ic.detach().cpu()),
                    'bc': float(L_bc.detach().cpu())
                })
                
                return loss

            for step in range(steps_lbfgs):
                loss = opt_lbfgs.step(closure)
                
                if step % 10 == 0:
                    # Get loss components from closure
                    parts = getattr(closure, 'loss_components', {'pde': 0, 'ic': 0, 'bc': 0})
                    hist.append((step + steps_adam, float(loss.item()), parts))
                    print(f"[L-BFGS {step:04d}] total={hist[-1][1]:.3e} parts={parts}")
                    
                    if plot_live:
                        self._plot_live_training(hist, tx_eval, x_eval, t_eval, 
                                               step=step+steps_adam, 
                                               total_steps=steps_adam + steps_lbfgs, 
                                               title_suffix=' - L-BFGS steps')

            if plot_live:
                self._plot_live_training(hist, tx_eval, x_eval, t_eval, 
                                       step=steps_adam + steps_lbfgs, 
                                       total_steps=steps_adam + steps_lbfgs, 
                                       title_suffix=' - Training ended')

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
        
        self.pred_hist.append(
            {'step' : step,
            'x' : x_eval.squeeze(),
            'pred' : u_pred.squeeze(), 
            }
        )
        
        fig, axarr = plt.subplots(1, 3, figsize=(12, 4))
        axarr[0].plot(x_eval.squeeze(), u_ref.squeeze(), label='Analytical True', lw=2)
        axarr[0].plot(x_eval.squeeze(), u_pred.squeeze(), '--', label='PINN', lw=2)
        for r, record in enumerate(self.pred_hist[-5:]):
            axarr[0].plot(record['x'], record['pred'], 'k-', alpha=0.2, lw=2)
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


class PINN_Wave1D_withLBFGS(PINN_Wave1D):
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
        super().__init__(model_arch, wave_speed, wave_mode, domain_length, samples_sizes, loss_weights, device)
        
    def train(self, 
            steps_adam=1500, lr=1e-3, 
            steps_lbfgs = 0,
            log_every=100, plot_live=False):
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
                    self._plot_live_training(hist, tx_eval, x_eval, t_eval, step=step, total_steps=steps_adam + steps_lbfgs, title_suffix = ' - Adam steps')

        self._plot_live_training(hist, tx_eval, x_eval, t_eval, step=step, total_steps=steps_adam + steps_lbfgs, title_suffix = ' - Adam steps')

        # L-BFGS refinement
        if steps_lbfgs > 0:
            print(f"Starting L-BFGS refinement for {steps_lbfgs} steps...")
            opt_lbfgs = torch.optim.LBFGS(
                self.model.parameters(), 
                lr=1.0,  # L-BFGS typically uses lr=1.0
                line_search_fn='strong_wolfe'
            )
            
            def closure():
                opt_lbfgs.zero_grad()

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

                # (4) Total loss
                loss = self.w_pde * L_pde + self.w_ic * L_ic + self.w_bc * L_bc
                loss.backward()
                
                # Store components for logging (closure can be called multiple times)
                if not hasattr(closure, 'loss_components'):
                    closure.loss_components = {}
                closure.loss_components.update({
                    'pde': float(L_pde.detach().cpu()),
                    'ic': float(L_ic.detach().cpu()),
                    'bc': float(L_bc.detach().cpu())
                })
                
                return loss

            for step in range(steps_lbfgs):
                loss = opt_lbfgs.step(closure)
                
                if step % 10 == 0:
                    # Get loss components from closure
                    parts = getattr(closure, 'loss_components', {'pde': 0, 'ic': 0, 'bc': 0})
                    hist.append((step + steps_adam, float(loss.item()), parts))
                    print(f"[L-BFGS {step:04d}] total={hist[-1][1]:.3e} parts={parts}")
                    
                    if plot_live:
                        self._plot_live_training(hist, tx_eval, x_eval, t_eval, 
                                               step=step+steps_adam, 
                                               total_steps=steps_adam + steps_lbfgs, 
                                               title_suffix=' - L-BFGS steps')

            if plot_live:
                self._plot_live_training(hist, tx_eval, x_eval, t_eval, 
                                       step=steps_adam + steps_lbfgs, 
                                       total_steps=steps_adam + steps_lbfgs, 
                                       title_suffix=' - Training ended')

        return hist


class PINN_Wave1D_AdaptiveLambdasKendall(PINN_Wave1D):
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
        super().__init__(model_arch, wave_speed, wave_mode, domain_length, samples_sizes, loss_weights, device)
        
        self.sigma2_pde = nn.Parameter(torch.tensor(float(loss_weights['pde']), device=self.device))
        self.sigma2_ic = nn.Parameter(torch.tensor(float(loss_weights['ic']), device=self.device))
        self.sigma2_bc = nn.Parameter(torch.tensor(float(loss_weights['bc']), device=self.device))
        self.list_sigmas = [self.sigma2_pde, self.sigma2_ic, self.sigma2_bc]

    def train(self, 
            steps_adam=1500, lr=1e-3, 
            steps_lbfgs = 0,
            log_every=100, plot_live=False):
        """Train the PINN model"""
        self.model.train()
        opt = torch.optim.Adam(
            list(self.model.parameters()) + self.list_sigmas, lr=lr)
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
            w_pde = 0.5 * torch.exp(- self.sigma2_pde) # 1/(2 σ^2)
            w_ic = 0.5 * torch.exp(- self.sigma2_ic)
            w_bc = 0.5 * torch.exp(- self.sigma2_bc)
            
            partial_pde = w_pde * L_pde + 0.5 * self.sigma2_pde # + (1/2) log σ^2
            partial_ic = w_ic * L_ic + 0.5 * self.sigma2_ic
            partial_bc = w_bc * L_bc + 0.5 * self.sigma2_bc

            loss = partial_pde + partial_ic + partial_bc
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
                    title_suffix = f"w_pde = {w_pde:.3f}, w_ic = {w_ic:.3f}, w_bc = {w_bc:.3f}"
                    self._plot_live_training(hist, tx_eval, x_eval, t_eval, step=step, total_steps=steps_adam + steps_lbfgs, title_suffix = title_suffix)

        if plot_live:
            title_suffix = f"w_pde = {w_pde:.3f}, w_ic = {w_ic:.3f}, w_bc = {w_bc:.3f}"
            self._plot_live_training(hist, tx_eval, x_eval, t_eval, 
                                step=steps_adam + steps_lbfgs, 
                                total_steps=steps_adam + steps_lbfgs, 
                                title_suffix=' - Training ended --> ' + title_suffix)

        print('Training ended.')
        print('Values of loss weigths:')
        print(f'-- w_pde: {w_pde:.3f}')
        print(f'-- w_ic: {w_ic:.3f}')
        print(f'-- w_bc: {w_bc:.3f}')

        return hist
    

class PINN_Wave1D_AdaptiveLambdasMcClenny(PINN_Wave1D):
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
        super().__init__(model_arch, wave_speed, wave_mode, domain_length, samples_sizes, loss_weights, device)
        
        # Initialize per-point weights (alpha parameters)
        self.alpha_pde = nn.Parameter(torch.ones(self.N_pde, 1, device=device))
        self.alpha_ic = nn.Parameter(torch.ones(self.N_ic * 2, 1, device=device))  # *2 for u and ut
        self.alpha_bc = nn.Parameter(torch.ones(self.N_bc * 2, 1, device=device))  # *2 for u and ux
        
        print(f"SA-PINN initialized with {self.N_pde} PDE points, {self.N_ic*2} IC points, {self.N_bc*2} BC points")

    def _norm_weights(self, alpha):
        # Softmax normalization: ensures positive weights that sum to N (mean ≈ 1)
        w = torch.softmax(alpha.flatten(), dim=0)
        return (w * alpha.numel()).view(-1, 1)

    def train(self, 
            steps_adam=1500, 
            lr=1e-3, lr_alpha=1e-3,
            steps_lbfgs=0,
            log_every=100, plot_live=False):
        """Train the PINN model with SA-PINN approach"""
        self.model.train()
        
        # Separate optimizers for model parameters and alpha weights
        opt_theta = torch.optim.Adam(self.model.parameters(), lr=lr)
        opt_alpha = torch.optim.Adam([self.alpha_pde, self.alpha_ic, self.alpha_bc], lr=lr_alpha)
        
        hist = []
        
        # Evaluation points for live plotting
        if plot_live:
            Nx = 400
            x_eval = torch.linspace(0, self.L, Nx)[:, None]
            t_eval = 0.35
            tx_eval = torch.cat([torch.full_like(x_eval, t_eval), x_eval], dim=1).to(self.device)

        for step in range(steps_adam):
            
            if step % 200 == 0:
                # optional resample (then re-init weights)
                tx_c = self.sample_collocation(self.N_pde).requires_grad_(True)
                tx_i, u0, ut0 = self.sample_ic(self.N_ic)
                tx_i.requires_grad_(True)
                tx0, txL = self.sample_bc(self.N_bc)
                tx0.requires_grad_(True)
                txL.requires_grad_(True)
                for a in (self.alpha_pde, self.alpha_ic, self.alpha_bc):
                    a.data.zero_()
            
            # Zero gradients
            opt_theta.zero_grad()
            opt_alpha.zero_grad()       

            # (1) PDE residuals
            u_c = self.model(tx_c)
            ut_c, ux_c, utt_c, uxx_c = self.derivatives(u_c, tx_c)
            res_pde = (utt_c - self.c**2 * uxx_c)  # Shape: [N_pde, 1]

            # (2) IC residuals
            tx_i = tx_i
            u_i = self.model(tx_i)
            ut_i, _, _, _ = self.derivatives(u_i, tx_i)
            res_ic = torch.cat([(u_i - u0), (ut_i - ut0)], dim=0)  # Shape: [2*N_ic, 1]

            # (3) BC residuals
            u0b = self.model(tx0)
            uLb = self.model(txL)
            _, ux0, _, _ = self.derivatives(u0b, tx0)
            _, uxL, _, _ = self.derivatives(uLb, txL)
            res_bc = torch.cat([(u0b - uLb), (ux0 - uxL)], dim=0)  # Shape: [2*N_bc, 1]

            # (4) Compute normalized weights
            w_pde = self._norm_weights(self.alpha_pde)  # Shape: [N_pde, 1]
            w_ic = self._norm_weights(self.alpha_ic)    # Shape: [2*N_ic, 1]
            w_bc = self._norm_weights(self.alpha_bc)    # Shape: [2*N_bc, 1]
            
            # (5) Weighted losses
            L_pde = torch.mean(w_pde * res_pde**2)
            L_ic = torch.mean(w_ic * res_ic**2)
            L_bc = torch.mean(w_bc * res_bc**2)

            total_loss = L_pde + L_ic + L_bc
            
            # (6) Backward pass
            total_loss.backward()

            # (7) Gradient ascent on alpha (max-min formulation)
            # Flip gradients for alpha parameters to perform gradient ascent
            for alpha_param in [w_pde, w_ic, w_bc]:
                if alpha_param.grad is not None:
                    alpha_param.grad.mul_(-1.0)

            # (8) Update parameters
            opt_theta.step()  # Gradient descent on θ
            opt_alpha.step()  # Gradient ascent on α (due to flipped gradients)

            if step % log_every == 0:
                parts = {
                    'pde': float(L_pde.detach().cpu()),
                    'ic': float(L_ic.detach().cpu()),
                    'bc': float(L_bc.detach().cpu())
                }
                hist.append((step, float(total_loss.detach().cpu()), parts))
                print(f"[SA-PINN {step:04d}] total={hist[-1][1]:.3e} parts={parts}")
                print(f"  Weight stats - PDE: {self.alpha_pde.mean():.3f}±{self.alpha_pde.std():.3f}, IC: {self.alpha_ic.mean():.3f}±{self.alpha_ic.std():.3f}, BC: {self.alpha_bc.mean():.3f}±{self.alpha_bc.std():.3f}")

                if plot_live:
                    title_suffix = f" SA-PINN: w_pde={self.alpha_pde.mean():.3f}, w_ic={self.alpha_ic.mean():.3f}, w_bc={self.alpha_bc.mean():.3f}"
                    self._plot_live_training(hist, tx_eval, x_eval, t_eval, step=step, total_steps=steps_adam + steps_lbfgs, title_suffix=title_suffix)

        if plot_live:
            self._plot_live_training(hist, tx_eval, x_eval, t_eval, 
                                step=steps_adam + steps_lbfgs, 
                                total_steps=steps_adam + steps_lbfgs, 
                                title_suffix=' - SA-PINN Training ended')

        return hist


class PINN_Wave1D_SamplingLatineHyperCube(PINN_Wave1D):
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
        super().__init__(model_arch, wave_speed, wave_mode, domain_length, samples_sizes, loss_weights, device)

    
    def latin_hypercube_sample(self, n_samples, n_dims, device='cpu'):
        """Generate Latin Hypercube samples using PyTorch"""
        # Generate stratified samples
        samples = torch.zeros(n_samples, n_dims, device=device)
        
        for dim in range(n_dims):
            # Create stratified grid
            intervals = torch.linspace(0, 1, n_samples + 1, device=device)
            # Add random jitter within each interval
            jitter = torch.rand(n_samples, device=device) / n_samples
            samples[:, dim] = intervals[:-1] + jitter
            # Random permutation to break correlation between dimensions
            samples[:, dim] = samples[torch.randperm(n_samples), dim]
        
        return samples
    
    def sample_collocation(self, Nc):
        """Sample collocation points in domain [0,1] x [0,L] using Latin Hypercube Sampling"""
        samples = self.latin_hypercube_sample(Nc, 2, self.device)
        
        # Scale to domain: t ∈ [0,1], x ∈ [0,L]
        t = samples[:, 0:1]  # Already in [0,1]
        x = samples[:, 1:2] * self.L  # Scale to [0,L]
        
        return torch.cat([t, x], dim=1)

    def sample_ic(self, Ni):
        """Sample initial condition points using Latin Hypercube Sampling"""
        x_samples = self.latin_hypercube_sample(Ni, 1, self.device)
        
        x = x_samples * self.L
        t = torch.zeros_like(x)
        u0 = torch.sin(self.k * x)
        ut0 = -self.omega * torch.cos(self.k * x)
        return torch.cat([t, x], 1), u0, ut0

    def sample_bc(self, Nb):
        """Sample boundary condition points using Latin Hypercube Sampling"""
        t_samples = self.latin_hypercube_sample(Nb, 1, self.device)
        
        t = t_samples
        x0 = torch.zeros_like(t)
        xL = torch.full_like(t, self.L)
        return (torch.cat([t, x0], 1), torch.cat([t, xL], 1))
    
    
class PINN_Wave1D_SamplingRAD(PINN_Wave1D):
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
        super().__init__(model_arch, wave_speed, wave_mode, domain_length, samples_sizes, loss_weights, device)
        

    def residual_based_sampling(self,
                                N_new,
                                pool_mult=10,
                                tau=1.0,
                                mix=0.1,
                                device=None):
        """
        Residual-based adaptive sampling (RS) for PINNs (Wu et al., 2023).
        Returns N_new collocation points biased toward large PDE residuals.

        Args:
            N_new (int): number of collocation points to return.
            pool_mult (int): candidate pool size multiplier (M = pool_mult * N_new).
            tau (float): tempering exponent; sample ∝ |residual|^tau (tau↑ => greedier).
            mix (float): fraction sampled uniformly for exploration (0..1).
            method (str): "prob" for probabilistic sampling, "topk" for greedy top-k.
            device: torch device (defaults to model device).

        Requires (already defined in your notebook):
            - model: PINN network mapping (t,x)->u
            - sample_collocation(N): uniform sampler in [0,1]^2 (scaled domain)
            - derivatives(u, tx): returns ut, ux, utt, uxx
            - c: wave speed (torch scalar)
        """
        assert 0.0 <= mix < 1.0
        M = int(pool_mult * N_new)
        k_uni  = int(round(mix * N_new))
        k_adpt = N_new - k_uni

        # 1) Candidate pool
        tx_pool = self._sample_collocation(M).requires_grad_(True)
        if device is not None:
            tx_pool = tx_pool.to(device)

        # 2) Residuals (no need to keep graph)
        u = self.model(tx_pool)
        ut, ux, utt, uxx = self.derivatives(u, tx_pool)  # your helper; uses create_graph=True internally
        r = (utt - self.c**2 * uxx).detach().squeeze()   # shape: (M,)

        # 3) Scores & selection
        scores = (r.abs() + 1e-12) ** float(tau)

        # probabilistic: sample ∝ scores
        probs = scores / scores.sum()
        # if all residuals ~0, fall back to uniform
        if torch.isnan(probs).any() or (probs.sum() <= 0):
            probs = torch.full_like(scores, 1.0 / M)
        idx_adpt = torch.multinomial(probs, k_adpt, replacement=False)

        # uniform exploration slice
        if k_uni > 0:
            idx_uni = torch.randint(0, M, (k_uni,), device=tx_pool.device)
            idx = torch.cat([idx_adpt, idx_uni], dim=0)
        else:
            idx = idx_adpt

        # 4) Return new collocation points
        return tx_pool[idx]
    
    
    def _sample_collocation(self, Nc):
        """Sample collocation points in domain [0,1] x [0,L]"""
        t = torch.rand(Nc, 1)
        x = torch.rand(Nc, 1) * self.L
        return torch.cat([t, x], dim=1).to(self.device)
    
    def sample_collocation(self, Nc):
        """Sample collocation points in domain [0,1] x [0,L] using Latin Hypercube Sampling"""
    
        # replace your uniform collocation sampling step with:
        tx_c = self.residual_based_sampling(
            N_new=int(Nc), pool_mult=10, tau=1.0, mix=0.1)
                
        return tx_c
    

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