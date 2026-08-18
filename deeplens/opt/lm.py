"""Levenberg-Marquardt Optimizer for DeepLens GeoLens optical systems with optional Paraxial ABCD Solve."""

import time
import logging
import torch
from typing import List, Tuple, Optional
from .paraxial import apply_paraxial_solve


class GeoLensLMOptimizer:
    """Levenberg-Marquardt (LM) damped Gauss-Newton optimizer for GeoLens.

    Minimizes the transverse ray aberration residual vector:
        r(theta) = [x_ray - x_ref, y_ray - y_ref]
    by solving the damped second-order normal equations at each iteration:
        (J^T J + lambda * diag(J^T J)) Delta = -J^T r

    Args:
        lens: GeoLens optical system instance.
        target_efl (float, optional): If provided, algebraically locks system EFL and BFL
            via differentiable ABCD paraxial solve on every candidate step. Defaults to None.
        solve_surf_idx (int): Surface index to solve curvature for target EFL. Defaults to -1.
        optim_mat (bool): Whether to optimize material refractive parameters.
        optim_surf_range (list): Surface indices to optimize. Defaults to all.
        lm_lambda (float): Initial damping parameter.
        lambda_up (float): Factor to increase lambda when a step is rejected.
        lambda_down (float): Factor to decrease lambda when a step is accepted.
        min_damp (float): Minimum diagonal damping value for numerical stability.
    """

    def __init__(
        self,
        lens,
        target_efl: Optional[float] = None,
        solve_surf_idx: int = -1,
        optim_mat: bool = False,
        optim_surf_range: Optional[List[int]] = None,
        lm_lambda: float = 0.1,
        lambda_up: float = 2.0,
        lambda_down: float = 3.0,
        min_damp: float = 1e-6,
    ):
        self.lens = lens
        self.target_efl = target_efl
        self.solve_surf_idx = solve_surf_idx
        self.optim_mat = optim_mat
        self.optim_surf_range = optim_surf_range
        self.lm_lambda = lm_lambda
        self.lambda_up = lambda_up
        self.lambda_down = lambda_down
        self.min_damp = min_damp

        if self.target_efl is not None:
            apply_paraxial_solve(self.lens, target_efl=self.target_efl, solve_surf_idx=self.solve_surf_idx)

        # Activate requires_grad on trainable parameters
        self.param_groups = self.lens.get_optimizer_params(
            optim_mat=optim_mat, optim_surf_range=optim_surf_range
        )

        # Collect scalar parameter pointers
        self.param_ptrs = self._collect_param_pointers()
        self.n_var = len(self.param_ptrs)

        logging.info(
            f"GeoLensLMOptimizer initialized with {self.n_var} active optical parameters (target_efl={self.target_efl})."
        )

    def _collect_param_pointers(self) -> List[Tuple]:
        """Extract explicit (object, attribute_name, index) pointers for all active parameters."""
        ptrs = []
        surfs_to_opt = (
            range(len(self.lens.surfaces))
            if self.optim_surf_range is None
            else self.optim_surf_range
        )
        
        num_surfs = len(self.lens.surfaces)
        actual_solve_idx = (num_surfs + self.solve_surf_idx) if self.solve_surf_idx < 0 else self.solve_surf_idx

        for idx in surfs_to_opt:
            surf = self.lens.surfaces[idx]
            # Surface distance / spacing d
            if hasattr(surf, "d") and isinstance(surf.d, torch.Tensor) and surf.d.requires_grad:
                ptrs.append((surf, "d", None))
            # Curvature c (exclude if actively solved paraxially)
            if hasattr(surf, "c") and isinstance(surf.c, torch.Tensor) and surf.c.requires_grad:
                if not (self.target_efl is not None and idx == actual_solve_idx):
                    ptrs.append((surf, "c", None))
            # Conic constant k
            if hasattr(surf, "k") and isinstance(surf.k, torch.Tensor) and surf.k.requires_grad:
                ptrs.append((surf, "k", None))
            # Aspheric coefficients ai
            if hasattr(surf, "ai") and isinstance(surf.ai, (list, torch.Tensor)):
                if isinstance(surf.ai, list):
                    for a_idx, a_val in enumerate(surf.ai):
                        if isinstance(a_val, torch.Tensor) and a_val.requires_grad:
                            ptrs.append((surf, "ai", a_idx))
                elif isinstance(surf.ai, torch.Tensor) and surf.ai.requires_grad:
                    for a_idx in range(surf.ai.numel()):
                        ptrs.append((surf, "ai", a_idx))

        # Sensor distance (exclude if actively solved paraxially)
        if (
            hasattr(self.lens, "d_sensor")
            and isinstance(self.lens.d_sensor, torch.Tensor)
            and self.lens.d_sensor.requires_grad
            and self.target_efl is None
        ):
            ptrs.append((self.lens, "d_sensor", None))

        return ptrs

    def get_param_value(self, obj, attr, idx) -> torch.Tensor:
        if idx is None:
            return getattr(obj, attr)
        return getattr(obj, attr)[idx]

    def set_param_value(self, obj, attr, idx, val: torch.Tensor):
        if idx is None:
            setattr(obj, attr, val)
        else:
            getattr(obj, attr)[idx] = val

    def get_flat_params(self) -> torch.Tensor:
        """Get flattened 1D parameter vector."""
        vals = [self.get_param_value(obj, attr, idx).detach().reshape(-1) for obj, attr, idx in self.param_ptrs]
        return torch.cat(vals)

    def set_flat_params(self, flat_vec: torch.Tensor):
        """Assign flattened 1D parameter vector to lens surfaces."""
        with torch.no_grad():
            for j, (obj, attr, idx) in enumerate(self.param_ptrs):
                val = flat_vec[j]
                if idx is None:
                    getattr(obj, attr).copy_(val)
                else:
                    getattr(obj, attr)[idx].copy_(val)
            if self.target_efl is not None:
                apply_paraxial_solve(self.lens, target_efl=self.target_efl, solve_surf_idx=self.solve_surf_idx)

    def compute_residuals(self, rays_list: list, pinhole_ref: torch.Tensor) -> torch.Tensor:
        """Compute transverse ray aberration vector across R, G, B wavelengths."""
        if self.target_efl is not None:
            apply_paraxial_solve(self.lens, target_efl=self.target_efl, solve_surf_idx=self.solve_surf_idx)

        res = []
        center_ref = None
        for wv_idx in [1, 0, 2]:  # green, red, blue
            ray = rays_list[wv_idx].clone()
            ray = self.lens.trace2sensor(ray)

            centroid_xy = ray.centroid()[..., :2]
            if center_ref is None:
                center_ref = centroid_xy.detach().unsqueeze(-2)

            ray_valid = ray.is_valid.bool().unsqueeze(-1)
            raw_err = ray.o[..., :2] - center_ref
            
            # Smooth clamped residual vector with restoring gradient for vignetted rays
            clean_err = torch.where(
                ray_valid,
                raw_err,
                torch.clamp(raw_err, -5.0, 5.0)
            ).reshape(-1)
            res.append(clean_err)

        return torch.cat(res)

    def step(
        self,
        rays_list: list,
        pinhole_ref: torch.Tensor,
        shape_control: bool = True,
    ) -> Tuple[float, bool]:
        """Execute one Levenberg-Marquardt iteration.

        Returns:
            loss (float): Current/new loss.
            accepted (bool): Whether the candidate step was accepted.
        """
        current_flat = self.get_flat_params()

        # 1. Current residuals & loss
        current_r = self.compute_residuals(rays_list, pinhole_ref)
        current_loss = 0.5 * (current_r ** 2).sum().item()

        # 2. Compute Jacobian J using forward-mode automatic differentiation
        J_cols = []
        for j, (obj, attr, idx) in enumerate(self.param_ptrs):
            orig_val = self.get_param_value(obj, attr, idx)
            with torch.autograd.forward_ad.dual_level():
                dual_val = torch.autograd.forward_ad.make_dual(
                    orig_val, torch.tensor(1.0, device=orig_val.device, dtype=orig_val.dtype)
                )
                self.set_param_value(obj, attr, idx, dual_val)
                r_dual = self.compute_residuals(rays_list, pinhole_ref)
                _, col_j = torch.autograd.forward_ad.unpack_dual(r_dual)
                self.set_param_value(obj, attr, idx, orig_val)
                if col_j is None:
                    col_j = torch.zeros_like(current_r)
                J_cols.append(col_j)

        J = torch.stack(J_cols, dim=1)  # Shape: [M, N]
        H = J.T @ J  # Gauss-Newton approximate Hessian: [N, N]
        g = J.T @ current_r  # Gradient: [N]

        # 3. Form damped Gauss-Newton system: (H + lambda * D) Delta = -g
        diag_H = torch.diag(H)
        diag_damp = torch.diag(torch.clamp(diag_H, min=self.min_damp))
        A = H + self.lm_lambda * diag_damp

        try:
            step_dir = torch.linalg.solve(A, -g)
        except Exception:
            step_dir = -g / (diag_H + self.lm_lambda + 1e-4)

        # 4. Evaluate candidate step
        candidate_flat = current_flat + step_dir
        self.set_flat_params(candidate_flat)
        if shape_control:
            self.lens.correct_shape()

        with torch.no_grad():
            cand_r = self.compute_residuals(rays_list, pinhole_ref)
            cand_loss = 0.5 * (cand_r ** 2).sum().item()

        actual_reduction = current_loss - cand_loss

        # 5. Adaptive Damping Step Update (Marquardt gain ratio)
        if actual_reduction > 0:
            # Step accepted
            self.lm_lambda = max(self.lm_lambda / self.lambda_down, 1e-7)
            return cand_loss, True
        else:
            # Step rejected: rollback parameters
            self.set_flat_params(current_flat)
            self.lm_lambda = min(self.lm_lambda * self.lambda_up, 1e7)
            return current_loss, False
