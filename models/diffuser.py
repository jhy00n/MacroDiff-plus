import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from diffusers import DDPMScheduler, DDIMScheduler
from models.model import MacroPlacer, TransPlacer, GraphPlacer
from utils.score import get_hpwl_norm_diff_test, get_hpwl_norm_diff, get_hpwl_grad, get_overlap_diff

torch.set_printoptions(profile="full")

class MacroDiff(nn.Module):
    def __init__(
        self,
        model,
        noise='full',
        timesteps=300,
        beta_schedule="linear",
        beta_start= 0.0001,
        beta_end=0.02,
        alpha=1.0,
        beta=1.0,
        max_iterations=500,
        guide_lr=0.001,
        weight_lr=0.01,
        threshold=0.1,
    ):
        super(MacroDiff, self).__init__()
        self.model = model
        self.noise = noise
        self.timesteps = timesteps
        self.alpha = alpha
        self.beta = beta

        self.scheduler = DDPMScheduler(
            num_train_timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule
        )

        self.sample_scheduler = DDPMScheduler(
            num_train_timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type='sample'
        )

        self.ddim_scheduler = DDIMScheduler(
            num_train_timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule
        )

        self.sample_ddim_scheduler = DDIMScheduler(
            num_train_timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type='sample'
        )

        num_inference_steps = self.timesteps // 10
        self.ddim_scheduler.set_timesteps(num_inference_steps)
        self.sample_ddim_scheduler.set_timesteps(num_inference_steps)

        self.max_iterations = max_iterations
        self.guide_lr = guide_lr
        self.weight_lr = weight_lr
        self.threshold = threshold

    ###################################################################################################################
    
    def pad_and_stack(self, sequences, max_len=None):
        if max_len == None:
            max_len = max(len(seq) for seq in sequences)
        padded = [torch.nn.functional.pad(seq, (0, 0, 0, max_len - len(seq)), value=0.) for seq in sequences]
        padded_tensor = torch.stack(padded)

        mask_list = []
        for seq in sequences:
            valid_len = len(seq)
            mask = torch.cat([
                torch.ones(valid_len, dtype=torch.bool, device=seq.device),
                torch.zeros(max_len - valid_len, dtype=torch.bool, device=seq.device)
            ])
            mask_list.append(mask)
        mask_tensor = torch.stack(mask_list)

        return padded_tensor, mask_tensor

    def process_chunks(self, data, split_sizes, num_io, batch_size):
        chunks = []
        start = 0
        for i in range(batch_size):
            end = split_sizes[i]
            chunk = data[start:end]
            chunks.append((chunk[:num_io[i]], chunk[num_io[i]:]))
            start = end
        return chunks

    def reset_pad(self, x, original_seqs, pad_value=0.):
        for i, seq in enumerate(original_seqs):
            length = seq.shape[0]
            x[i, length:] = pad_value
        return x
    
    def restore_pad_and_stack(self, x, num_io, num_macro):
        max_io = max(num_io)
        max_macro = max(num_macro)
        batch_size = len(num_io)
        restored_batches = [
            torch.cat([
                x[i, :max_io][:num_io[i]],
                x[i, max_io:][:num_macro[i]]
            ], dim=0) for i in range(batch_size)
        ]
        return torch.cat(restored_batches, dim=0)

    def forward(self, data):
        num_io = data['num_io']
        num_macro = data['num_macro']
        num_net = data['num_net']
        max_io = max(data['max_io'])
        max_macro = max(data['max_macro'])
        batch_size = len(num_io)

        size = data['cell']['size'].clone()
        pos = data['cell']['pos'].clone()
        degree = data['net']['degree'].clone()

        split_sizes = torch.cumsum(num_io + num_macro, dim=0)
        chunks = self.process_chunks(pos, split_sizes, num_io, batch_size)
        io_list, x0_list = zip(*chunks)
        io, io_mask = self.pad_and_stack(io_list, max_io)                      # batch_size X max_io X 2
        x0, x0_mask = self.pad_and_stack(x0_list, max_macro)                   # batch_size X max_macro X 2
        pos0 = torch.cat([io, x0], dim=1)                                       # batch_size X max_io + max_macro X 2
        is_macro = torch.cat([torch.zeros_like(io_mask, dtype=torch.float), x0_mask.float()], dim=1).unsqueeze(-1)
        ##################################################

        size_chunks = self.process_chunks(size, split_sizes, num_io, batch_size)
        io_size_list, x_size_list = zip(*size_chunks)
        io_size, _ = self.pad_and_stack(io_size_list, max_io)                      # batch_size X max_io X 2
        x_size, _ = self.pad_and_stack(x_size_list, max_macro)                     # batch_size X max_macro X 2
        size_pad = torch.cat([io_size, x_size], dim=1)                              # batch_size X max_io + max_macro X 2
        ##################################################
        
        _, d0 = get_hpwl_norm_diff(data, pos0)          # num_net X 1
        d0_pad, d0_mask = self.pad_and_stack(d0)           # batch_size X max_net X 1
    
        t = torch.randint(0, self.timesteps, (batch_size,), device=x0.device)     # Extract t

        t_io = t.unsqueeze(-1) * io_mask
        t_macro = t.unsqueeze(-1) * x0_mask
        t_cell_pad = torch.cat([t_io, t_macro], dim=1)

        t_net_pad = t.unsqueeze(-1) * d0_mask

        t_cell = self.restore_pad_and_stack(t_cell_pad, num_io, num_macro)
        t_net = self.restore_pad_and_stack(t_net_pad, num_net, [0]*len(num_net))

        noise = torch.randn_like(x0)                                     # Random noise
        xt = self.scheduler.add_noise(x0, noise, t)                      # Add the noise into macro position
        xt = self.reset_pad(xt, x0_list)
        
        post = torch.cat([io, xt], dim=1)

        _, dt = get_hpwl_norm_diff(data, post)
        dt_pad, _ = self.pad_and_stack(dt)

        data['cell'].x = torch.cat([self.restore_pad_and_stack(post, num_io, num_macro), size, self.restore_pad_and_stack(is_macro, num_io, num_macro)], dim=1)
        data['net'].x = torch.cat([self.restore_pad_and_stack(dt_pad, num_net, [0]*len(num_net)), degree], dim=1)

        if isinstance(self.model, MacroPlacer):
            pred_net, pred_cell = self.model(data, t_cell, t_net, t, x_size, batch_size, max_macro, split_sizes, num_io, x0_mask)
        else:
            pred_net, pred_cell = self.model(data, t_cell, t_net, post[:,-max_macro:], t, size_pad[:,-max_macro:], x0_mask)
        x0_mask_expanded = x0_mask.unsqueeze(-1).expand(-1, -1, 2).float()

        if not isinstance(self.model, TransPlacer):
            with torch.no_grad():
                hpwl_grad = get_hpwl_grad(data, post)
            net_split_sizes = torch.cumsum(num_net, dim=0)
            net_chunks = self.process_chunks(pred_net, net_split_sizes, [0] * batch_size, batch_size)
            pred_net_pad, _ = self.pad_and_stack([chunk[1] for chunk in net_chunks])
        else:
            pred_net_pad = None

        if isinstance(self.model, GraphPlacer):
            pred_chuncks = self.process_chunks(pred_cell, split_sizes, num_io, batch_size)
            pred_io_list, pred_cell_list = zip(*pred_chuncks)
            pred_cell_pad, _ = self.pad_and_stack(pred_cell_list, max_macro)
        else:
            pred_cell_pad = pred_cell

        loss = 0.0
        cost = 0.0
        metric = 0.0
        ol_list = []
        pred_noise_list = []
        for i in range(batch_size):
            if self.noise == 'full':
                cell_noise = torch.matmul(hpwl_grad[i][-num_macro[i]:], pred_net_pad[i,:num_net[i]]).squeeze()
                pred_noise_step = self.alpha * cell_noise + self.beta * pred_cell_pad[i,:num_macro[i]]
            elif self.noise == 'cell':
                pred_noise_step = pred_cell_pad[i,:num_macro[i]]
            elif self.noise == 'net':
                pred_noise_step = torch.matmul(hpwl_grad[i][-num_macro[i]:], pred_net_pad[i,:num_net[i]]).squeeze()
            else:
                raise ValueError(f"Invalid noise: {self.noise}")

            pred_noise_list.append(pred_noise_step)

        pred_noise, _ = self.pad_and_stack(pred_noise_list, max_macro)

        loss = F.mse_loss(pred_noise, noise, reduction='none')
        loss = loss * x0_mask_expanded
        
        loss = loss.sum() / x0_mask_expanded.sum()

        return loss
    
    def valid(self, data, eta=1.0, seed=None):
        with torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)
            else:
                torch.manual_seed(0)

            num_io = data['num_io']
            num_macro = data['num_macro']

            size = data['cell']['size'].clone()
            pos = data['cell']['pos'].clone()
            degree = data['net']['degree'].clone()

            io, x0 = pos.split([num_io, num_macro])
            is_macro = torch.cat([
                torch.zeros(io.size(0), 1, dtype=torch.float, device=io.device), 
                torch.ones(x0.size(0), 1, dtype=torch.float, device=x0.device)
            ], dim=0)

            xt = torch.randn_like(x0)
            x_list = [xt.clone()]

        for t in tqdm(self.scheduler.timesteps, desc = 'sampling loop time step', total=len(self.scheduler.timesteps)):
            with torch.no_grad():
                t_tensor = torch.tensor([t], device=xt.device)
                post = torch.cat([io, xt], dim=0)
                _, dt = get_hpwl_norm_diff_test(data, post)

                data['cell'].x = torch.cat([post, size, is_macro], dim=1)
                data['net'].x = torch.cat([dt, degree], dim=1)
                t_cell = t_tensor.expand(data['cell'].x.shape[0])
                t_net = t_tensor.expand(data['net'].x.shape[0])

                xt_model_input = xt.unsqueeze(0)
                macro_size_model_input = size[num_io:].unsqueeze(0)

                if isinstance(self.model, MacroPlacer):
                    pred_net, pred_cell = self.model(data, t_cell, t_net, t_tensor, macro_size_model_input, 1, num_macro)
                else:
                    pred_net, pred_cell = self.model(data, t_cell, t_net, xt_model_input, t_tensor, macro_size_model_input)

            with torch.enable_grad():
                xt_for_grad = xt.detach().requires_grad_(True)
                post_for_grad = torch.cat([io, xt_for_grad], dim=0)

                if not isinstance(self.model, TransPlacer):
                    _, dt_for_grad = get_hpwl_norm_diff_test(data, post_for_grad)
                    dt_vjp_xt = torch.autograd.grad(
                        outputs=dt_for_grad,
                        inputs=xt_for_grad,
                        grad_outputs=pred_net,
                        retain_graph=False
                    )[0]
                else:
                    dt_vjp_xt = None

            with torch.no_grad():
                if self.noise == 'full':
                    pred_noise = self.alpha * dt_vjp_xt + self.beta * pred_cell[-num_macro:].squeeze()
                elif self.noise == 'cell':
                    pred_noise = pred_cell[-num_macro:].squeeze()
                elif self.noise == 'net':
                    pred_noise = dt_vjp_xt
                else:
                    raise ValueError(f"Invalid noise: {self.noise}")

            step_output = self.scheduler.step(pred_noise, t, xt)
            xt = step_output.prev_sample.to(xt.device)
            x_list.append(xt.clone())
            
        return xt.detach(), x_list

    @torch.enable_grad()
    def guided_valid(self, data, guide_x0=True, seed=None, w_hpwl=0.001, w_overlap=1.0, cluster=False, data_cluster=None, verbose_guidance=False):
        if seed is not None:
            torch.manual_seed(seed)
        else:
            torch.manual_seed(0)

        
        num_macro = data['num_macro']
        if cluster:
            num_cluster = data_cluster['num_cluster']
            num_io = data_cluster['cell']['pos'].shape[0] - num_macro - num_cluster
            size = data_cluster['cell']['size'].clone()
            pos = data_cluster['cell']['pos'].clone()
        else:
            num_cluster = 0
            num_io = data['num_io']
            size = data['cell']['size'].clone()
            pos = data['cell']['pos'].clone()
        degree = data['net']['degree'].clone()

        # Conditional splitting based on cluster mode
        if cluster and num_cluster > 0:
            io, x0, _ = pos.split([num_io, num_macro, num_cluster])
        else:
            io, x0 = pos.split([num_io, num_macro])
        is_macro = torch.cat([
            torch.zeros(io.size(0), 1, dtype=torch.float, device=io.device), 
            torch.ones(x0.size(0), 1, dtype=torch.float, device=x0.device)
        ], dim=0)

        xt = torch.randn_like(x0)
        x_list = [xt.clone()]

        for t in tqdm(self.ddim_scheduler.timesteps, desc = 'sampling loop time step'):
            with torch.no_grad():
                t_tensor = torch.tensor([t], device=xt.device)
                post = torch.cat([io, xt], dim=0)
                _, dt = get_hpwl_norm_diff_test(data, post)

                data['cell'].x = torch.cat([post, size[:num_io+num_macro], is_macro], dim=1)
                data['net'].x = torch.cat([dt, degree], dim=1)
                t_cell = t_tensor.expand(data['cell'].x.shape[0])
                t_net = t_tensor.expand(data['net'].x.shape[0])

                xt_model_input = xt.unsqueeze(0)
                macro_size_model_input = size[num_io:num_io+num_macro].unsqueeze(0)

                if isinstance(self.model, MacroPlacer):
                    pred_net, pred_cell = self.model(data, t_cell, t_net, t_tensor, macro_size_model_input, 1, num_macro)
                else:
                    pred_net, pred_cell = self.model(data, t_cell, t_net, xt_model_input, t_tensor, macro_size_model_input)

            if not isinstance(self.model, TransPlacer):
                with torch.no_grad():
                    hpwl_grad = get_hpwl_grad(data, post.unsqueeze(0))[0]
            else:
                hpwl_grad = None

            with torch.no_grad():
                if self.noise == 'full':
                    pred_noise = self.alpha * torch.matmul(hpwl_grad[-num_macro:], pred_net).squeeze() + self.beta * pred_cell[-num_macro:].squeeze()
                elif self.noise == 'cell':
                    pred_noise = pred_cell[-num_macro:].squeeze()
                elif self.noise == 'net':
                    pred_noise = torch.matmul(hpwl_grad[-num_macro:], pred_net)
                else:
                    raise ValueError(f"Invalid noise: {self.noise}")
                    
                step_output = self.ddim_scheduler.step(pred_noise, t, xt)
                xt = step_output.prev_sample.to(xt.device)
                x_hat_0 = step_output.pred_original_sample.to(xt.device)

            ###############################################################################
            if guide_x0:
                if cluster:
                    x_hat_0_guided = self.apply_guidance(
                        x_hat_0,
                        io,
                        size,
                        data_cluster,
                        num_io,
                        num_cluster,
                        w_hpwl=w_hpwl,
                        w_overlap=w_overlap
                    )
                else:
                    x_hat_0_guided = self.apply_guidance(
                        x_hat_0,
                        io,
                        size,
                        data,
                        num_io,
                        0,
                        w_hpwl=w_hpwl,
                        w_overlap=w_overlap
                    )

                with torch.no_grad():
                    if verbose_guidance:
                        print(f"x_hat_0_guided: {x_hat_0_guided.shape}, xt: {xt.shape}, t: {t}")
                    guided_step_output = self.sample_ddim_scheduler.step(x_hat_0_guided, t, xt)
                    xt = guided_step_output.prev_sample.to(xt.device)
            else:
                # Apply guidance directly to xt (not x_hat_0)
                xt = self.apply_guidance(
                    x_hat_0,
                    io,
                    size,
                    data_cluster,
                    num_io,
                    num_cluster,
                    w_hpwl=w_hpwl,
                    w_overlap=w_overlap
                )
            ###############################################################################
                
            x_list.append(xt.clone())
            
        return xt.detach(), x_list

    def apply_guidance(self, x_hat_0, io, size, data, num_io, num_cluster, w_hpwl=0.01, w_overlap=10.0):
        # Only create and concatenate cluster tensor if clusters exist
        if num_cluster > 0:
            cluster = torch.zeros([num_cluster, 2], device=x_hat_0.device)
            x_k = torch.cat([x_hat_0.detach().clone(), cluster]).requires_grad_(True)
        else:
            x_k = x_hat_0.detach().clone().requires_grad_(True)
        macro_size = size[num_io:num_io+len(x_k)]
        optimizer = torch.optim.Adam([x_k], lr=self.guide_lr)

        # --- State Tracking for Dynamic Phase Switching ---
        phase = "HPWL_OPTIMIZATION"
        prev_loss_hpwl = float('inf')
        hpwl_improvement_threshold = 1e-5 # Threshold to detect HPWL convergence
        burn_in_iterations = 50 # Minimum iterations before switching phase

        for i in range(self.max_iterations):
            optimizer.zero_grad()

            # --- Calculate Losses ---
            pos_k = torch.cat([io, x_k], dim=0)
            loss_hpwl, _ = get_hpwl_norm_diff_test(data, pos_k)
            loss_overlap, _ = get_overlap_diff(macro_size, x_k)

            # --- Dynamic Phase Transition Logic ---
            if phase == "HPWL_OPTIMIZATION" and i > burn_in_iterations:
                hpwl_improvement = prev_loss_hpwl - loss_hpwl.item()
                if hpwl_improvement < hpwl_improvement_threshold:
                    phase = "OVERLAP_REMOVAL"
                      
            prev_loss_hpwl = loss_hpwl.item()

            # --- Set weights based on the current phase ---
            if phase == "HPWL_OPTIMIZATION":
                # Focus on HPWL, with a small penalty for overlap
                w_hpwl = 0.01
                w_overlap = 10.0
            else: # OVERLAP_REMOVAL phase
                # Focus on removing overlap, with a small penalty for HPWL
                w_hpwl = 0.001
                w_overlap = 10.0

            # --- Calculate Total Loss ---
            total_loss = (w_hpwl * loss_hpwl + 
                          w_overlap * loss_overlap) # +
                          #w_move * loss_move)
            
            # --- Optimization Step ---
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(x_k, max_norm=1.0)
            optimizer.step()
            
            # --- Clamp Positions to Die Area ---
            x_k.data = torch.clamp(x_k.data, min=-1.0 + torch.zeros_like(macro_size), max=1.0 - macro_size)

        # Conditional return based on cluster existence
        if num_cluster > 0:
            return x_k[:-num_cluster].detach().clone()
        else:
            return x_k.detach().clone()

