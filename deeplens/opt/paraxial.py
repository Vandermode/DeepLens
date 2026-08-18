"""Differentiable ABCD Paraxial Optical Solver for DeepLens GeoLens systems."""

import torch
from typing import Tuple, Optional


def compute_system_abcd(lens) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the 2x2 paraxial ABCD transfer matrix of the lens system in reduced coordinates.

    Args:
        lens: GeoLens instance.

    Returns:
        efl (torch.Tensor): Effective focal length in mm.
        bfl (torch.Tensor): Back focal length in mm.
        M_total (torch.Tensor): 2x2 paraxial transfer matrix.
    """
    device = lens.device if hasattr(lens, "device") else torch.device("cpu")
    n_curr = 1.0
    mats = []

    for i, s in enumerate(lens.surfaces):
        n_next = float(s.mat2.n) if hasattr(s.mat2, "n") else 1.0
        c = s.c if hasattr(s, "c") and isinstance(s.c, torch.Tensor) else torch.tensor(0.0, device=device)
        dn = n_next - n_curr

        # Refraction matrix R: [[1, 0], [-c * dn, 1]]
        P = c * dn
        R = torch.stack([
            torch.stack([torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)]),
            torch.stack([-P, torch.tensor(1.0, device=device)])
        ])
        mats.append(R)

        # Propagation matrix to next surface
        if i < len(lens.surfaces) - 1:
            t = lens.surfaces[i + 1].d - s.d
            T = torch.stack([
                torch.stack([torch.tensor(1.0, device=device), t / n_next]),
                torch.stack([torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)])
            ])
            mats.append(T)

        n_curr = n_next

    # Accumulate total system matrix from object side to image side
    M_total = torch.eye(2, dtype=torch.float32, device=device)
    for M in mats:
        M_total = M @ M_total

    A = M_total[0, 0]
    C = M_total[1, 1]  # In reduced form: C is power
    C_p = M_total[1, 0]

    efl = -1.0 / C_p if torch.abs(C_p) > 1e-8 else torch.tensor(float("inf"), device=device)
    bfl = A / (-C_p) if torch.abs(C_p) > 1e-8 else torch.tensor(float("inf"), device=device)
    return efl, bfl, M_total


def solve_paraxial_efl(
    lens,
    target_efl: float,
    solve_surf_idx: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve for the exact curvature c_k and sensor distance to enforce target EFL analytically.

    Given a system with target effective focal length (EFL), solves in closed-form:
        c_k = (1/(EFL * A_1 * D_2) + C_2/D_2 + C_1/A_1) / delta_n
        d_sensor = d_last + BFL

    Args:
        lens: GeoLens instance.
        target_efl (float): Desired focal length in mm.
        solve_surf_idx (int): Surface index whose curvature is solved. Defaults to -1 (last surface).

    Returns:
        solved_c (torch.Tensor): Solved curvature for surface k in 1/mm.
        solved_bfl (torch.Tensor): Solved back focal length in mm.
        solved_d_sensor (torch.Tensor): Solved absolute sensor position in mm.
    """
    device = lens.device if hasattr(lens, "device") else torch.device("cpu")
    num_surfs = len(lens.surfaces)
    if solve_surf_idx < 0:
        solve_surf_idx = num_surfs + solve_surf_idx

    n_curr = 1.0
    mats_left = []
    mats_right = []
    delta_n = None

    for i, s in enumerate(lens.surfaces):
        n_next = float(s.mat2.n) if hasattr(s.mat2, "n") else 1.0
        c = s.c if hasattr(s, "c") and isinstance(s.c, torch.Tensor) else torch.tensor(0.0, device=device)
        dn = n_next - n_curr

        if i == solve_surf_idx:
            delta_n = dn
        else:
            P = c * dn
            R = torch.stack([
                torch.stack([torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)]),
                torch.stack([-P, torch.tensor(1.0, device=device)])
            ])
            if i < solve_surf_idx:
                mats_left.append(R)
            else:
                mats_right.append(R)

        if i < num_surfs - 1:
            t = lens.surfaces[i + 1].d - s.d
            T = torch.stack([
                torch.stack([torch.tensor(1.0, device=device), t / n_next]),
                torch.stack([torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)])
            ])
            if i < solve_surf_idx:
                mats_left.append(T)
            else:
                mats_right.append(T)

        n_curr = n_next

    # Accumulate left and right sub-matrices
    M_left = torch.eye(2, dtype=torch.float32, device=device)
    for M in mats_left:
        M_left = M @ M_left

    M_right = torch.eye(2, dtype=torch.float32, device=device)
    for M in mats_right:
        M_right = M @ M_right

    A1, B1 = M_left[0, 0], M_left[0, 1]
    C1, D1 = M_left[1, 0], M_left[1, 1]
    A2, B2 = M_right[0, 0], M_right[0, 1]
    C2, D2 = M_right[1, 0], M_right[1, 1]

    # Analytic curvature solve
    t_efl = torch.tensor(float(target_efl), device=device)
    solved_c = (1.0 / (t_efl * A1 * D2) + C2 / D2 + C1 / A1) / delta_n

    # Total matrix with solved curvature
    P_solved = solved_c * delta_n
    R_solved = torch.stack([
        torch.stack([torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)]),
        torch.stack([-P_solved, torch.tensor(1.0, device=device)])
    ])
    M_tot = M_right @ R_solved @ M_left
    A_tot = M_tot[0, 0]
    C_tot = M_tot[1, 0]

    solved_bfl = -A_tot / C_tot
    solved_d_sensor = lens.surfaces[-1].d + solved_bfl

    return solved_c, solved_bfl, solved_d_sensor


def apply_paraxial_solve(lens, target_efl: float, solve_surf_idx: int = -1) -> None:
    """In-place update of lens curvature and sensor distance to enforce target EFL and focus."""
    with torch.no_grad():
        solved_c, _, solved_d_sensor = solve_paraxial_efl(
            lens, target_efl=target_efl, solve_surf_idx=solve_surf_idx
        )
        if solve_surf_idx < 0:
            solve_surf_idx = len(lens.surfaces) + solve_surf_idx
        
        surf = lens.surfaces[solve_surf_idx]
        if hasattr(surf, "c") and isinstance(surf.c, torch.Tensor):
            surf.c.copy_(solved_c)
        if hasattr(lens, "d_sensor") and isinstance(lens.d_sensor, torch.Tensor):
            lens.d_sensor.copy_(solved_d_sensor)
