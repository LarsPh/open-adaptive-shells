from pathlib import Path
# Modified for Adaptive Shells training and rendering; see the repository NOTICE.
from typing import Optional

import numpy as np
import pymeshlab
import torch
import torch.nn.functional as F
import trimesh
from skimage import measure
from tqdm import tqdm
avg_pool_3d = torch.nn.AvgPool3d(2, stride=2)
upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")
max_pool_3d = torch.nn.MaxPool3d(3, stride=1, padding=1)

from nerfstudio.fields.adashell_field import transform_inv_s

@torch.no_grad()
def get_surface_sliding(
    sdf,
    resolution=512,
    bounding_box_min=(-1.0, -1.0, -1.0),
    bounding_box_max=(1.0, 1.0, 1.0),
    return_mesh=False,
    level=0,
    coarse_mask=None,
    output_path: Path = Path("test.ply"),
    simplify_mesh=True,
):
    assert resolution % 512 == 0
    if coarse_mask is not None:
        # we need to permute here as pytorch's grid_sample use (z, y, x)
        coarse_mask = coarse_mask.permute(2, 1, 0)[None, None].cuda().float()

    resN = resolution
    cropN = 512
    level = 0
    N = resN // cropN

    grid_min = bounding_box_min
    grid_max = bounding_box_max
    xs = np.linspace(grid_min[0], grid_max[0], N + 1)
    ys = np.linspace(grid_min[1], grid_max[1], N + 1)
    zs = np.linspace(grid_min[2], grid_max[2], N + 1)

    # print(xs)
    # print(ys)
    # print(zs)
    meshes = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                # print(i, j, k)
                x_min, x_max = xs[i], xs[i + 1]
                y_min, y_max = ys[j], ys[j + 1]
                z_min, z_max = zs[k], zs[k + 1]

                x = np.linspace(x_min, x_max, cropN)
                y = np.linspace(y_min, y_max, cropN)
                z = np.linspace(z_min, z_max, cropN)

                xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
                points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()

                def evaluate(points):
                    z = []
                    for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
                        z.append(sdf(pnts))
                    z = torch.cat(z, axis=0)
                    return z

                # construct point pyramids
                points = points.reshape(cropN, cropN, cropN, 3).permute(3, 0, 1, 2)
                if coarse_mask is not None:
                    # breakpoint()
                    points_tmp = points.permute(1, 2, 3, 0)[None].cuda()
                    current_mask = torch.nn.functional.grid_sample(coarse_mask, points_tmp)
                    current_mask = (current_mask > 0.0).cpu().numpy()[0, 0]
                else:
                    current_mask = None

                points_pyramid = [points]
                for _ in range(3):
                    points = avg_pool_3d(points[None])[0]
                    points_pyramid.append(points)
                points_pyramid = points_pyramid[::-1]

                # evalute pyramid with mask
                mask = None
                threshold = 2 * (x_max - x_min) / cropN * 8
                for pid, pts in enumerate(points_pyramid):
                    coarse_N = pts.shape[-1]
                    pts = pts.reshape(3, -1).permute(1, 0).contiguous()

                    if mask is None:
                        # only evaluate
                        if coarse_mask is not None:
                            pts_sdf = torch.ones_like(pts[:, 1])
                            valid_mask = (
                                torch.nn.functional.grid_sample(coarse_mask, pts[None, None, None])[0, 0, 0, 0] > 0
                            )
                            if valid_mask.any():
                                pts_sdf[valid_mask] = evaluate(pts[valid_mask].contiguous())
                        else:
                            pts_sdf = evaluate(pts)
                    else:
                        mask = mask.reshape(-1)
                        pts_to_eval = pts[mask]

                        if pts_to_eval.shape[0] > 0:
                            pts_sdf_eval = evaluate(pts_to_eval.contiguous())
                            pts_sdf[mask] = pts_sdf_eval
                        # print("ratio", pts_to_eval.shape[0] / pts.shape[0])

                    if pid < 3:
                        # update mask
                        mask = torch.abs(pts_sdf) < threshold
                        mask = mask.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                        mask = upsample(mask.float()).bool()

                        pts_sdf = pts_sdf.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                        pts_sdf = upsample(pts_sdf)
                        pts_sdf = pts_sdf.reshape(-1)

                    threshold /= 2.0

                z = pts_sdf.detach().cpu().numpy()

                # skip if no surface found
                if current_mask is not None:
                    valid_z = z.reshape(cropN, cropN, cropN)[current_mask]
                    if valid_z.shape[0] <= 0 or (np.min(valid_z) > level or np.max(valid_z) < level):
                        continue

                if not (np.min(z) > level or np.max(z) < level):
                    z = z.astype(np.float32)
                    verts, faces, normals, _ = measure.marching_cubes(
                        volume=z.reshape(cropN, cropN, cropN),  # .transpose([1, 0, 2]),
                        level=level,
                        spacing=(
                            (x_max - x_min) / (cropN - 1),
                            (y_max - y_min) / (cropN - 1),
                            (z_max - z_min) / (cropN - 1),
                        ),
                        mask=current_mask,
                    )
                    # print(np.array([x_min, y_min, z_min]))
                    # print(verts.min(), verts.max())
                    verts = verts + np.array([x_min, y_min, z_min])
                    # print(verts.min(), verts.max())

                    meshcrop = trimesh.Trimesh(verts, faces, normals)
                    # meshcrop.export(f"{i}_{j}_{k}.ply")
                    meshes.append(meshcrop)

    combined = trimesh.util.concatenate(meshes)
    combined = clean_extracted_mesh(combined)

    if return_mesh:
        return combined
    else:
        filename = str(output_path)
        filename_simplify = str(output_path).replace(".ply", "-simplify.ply")
        combined.export(filename)
        if simplify_mesh:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(filename)

            print("simply mesh")
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=2000000)
            ms.save_current_mesh(filename_simplify, save_face_color=False)


@torch.no_grad()
def get_surface_occupancy(
    occupancy_fn,
    resolution=512,
    bounding_box_min=(-1.0, -1.0, -1.0),
    bounding_box_max=(1.0, 1.0, 1.0),
    return_mesh=False,
    level=0.5,
    device=None,
    output_path: Path = Path("test.ply"),
):
    grid_min = bounding_box_min
    grid_max = bounding_box_max
    N = resolution
    xs = np.linspace(grid_min[0], grid_max[0], N)
    ys = np.linspace(grid_min[1], grid_max[1], N)
    zs = np.linspace(grid_min[2], grid_max[2], N)

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).to(device=device)

    def evaluate(points):
        z = []
        for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
            z.append(occupancy_fn(pnts.contiguous()).contiguous())
        z = torch.cat(z, axis=0)
        return z

    z = evaluate(points).detach().cpu().numpy()

    if not (np.min(z) > level or np.max(z) < level):
        verts, faces, normals, _ = measure.marching_cubes(
            volume=z.reshape(resolution, resolution, resolution),
            level=level,
            spacing=(
                (grid_max[0] - grid_min[0]) / (N - 1),
                (grid_max[1] - grid_min[1]) / (N - 1),
                (grid_max[2] - grid_min[2]) / (N - 1),
            ),
        )
        verts = verts + np.array(grid_min)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        meshexport = trimesh.Trimesh(verts, faces, normals)
        meshexport.export(str(output_path))
    else:
        print("=================================================no surface skip")


def get_surface_sliding_with_contraction(
    sdf,
    resolution=512,
    bounding_box_min=(-1.0, -1.0, -1.0),
    bounding_box_max=(1.0, 1.0, 1.0),
    return_mesh=False,
    level=0,
    coarse_mask=None,
    output_path: Path = Path("test.ply"),
    simplify_mesh=True,
    inv_contraction=None,
    max_range=32.0,
):
    assert resolution % 512 == 0

    resN = resolution
    cropN = 512
    level = 0
    N = resN // cropN

    grid_min = bounding_box_min
    grid_max = bounding_box_max
    xs = np.linspace(grid_min[0], grid_max[0], N + 1)
    ys = np.linspace(grid_min[1], grid_max[1], N + 1)
    zs = np.linspace(grid_min[2], grid_max[2], N + 1)

    # print(xs)
    # print(ys)
    # print(zs)
    meshes = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                print(i, j, k)
                x_min, x_max = xs[i], xs[i + 1]
                y_min, y_max = ys[j], ys[j + 1]
                z_min, z_max = zs[k], zs[k + 1]

                x = np.linspace(x_min, x_max, cropN)
                y = np.linspace(y_min, y_max, cropN)
                z = np.linspace(z_min, z_max, cropN)

                xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
                points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()

                @torch.no_grad()
                def evaluate(points):
                    z = []
                    for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
                        z.append(sdf(pnts))
                    z = torch.cat(z, axis=0)
                    return z

                # construct point pyramids
                points = points.reshape(cropN, cropN, cropN, 3)

                # query coarse grids
                points_tmp = points[None].cuda() * 0.5  # normalize from [-2, 2] to [-1, 1]
                current_mask = torch.nn.functional.grid_sample(coarse_mask, points_tmp)

                points = points.reshape(-1, 3)
                valid_mask = current_mask.reshape(-1) > 0
                pts_to_eval = points[valid_mask]
                print(current_mask.float().mean())

                pts_sdf = torch.ones_like(points[..., 0]) * 100.0
                print(pts_sdf.shape, pts_to_eval.shape, points.shape)
                if pts_to_eval.shape[0] > 0:
                    pts_sdf_eval = evaluate(pts_to_eval.contiguous())
                    pts_sdf[valid_mask.reshape(-1)] = pts_sdf_eval

                # use min_pooling to remove masked marching cube artefacts
                min_sdf = max_pool_3d(pts_sdf.reshape(1, 1, cropN, cropN, cropN) * -1.0) * -1.0
                min_mask = (current_mask > 0.0).float()
                pts_sdf = pts_sdf.reshape(1, 1, cropN, cropN, cropN) * min_mask + min_sdf * (1.0 - min_mask)

                z = pts_sdf.detach().cpu().numpy()

                current_mask = (current_mask > 0.0).cpu().numpy()[0, 0]
                # skip if no surface found
                if current_mask is not None:
                    valid_z = z.reshape(cropN, cropN, cropN)[current_mask]
                    if valid_z.shape[0] <= 0 or (np.min(valid_z) > level or np.max(valid_z) < level):
                        continue

                if not (np.min(z) > level or np.max(z) < level):
                    z = z.astype(np.float32)
                    verts, faces, normals, _ = measure.marching_cubes(
                        volume=z.reshape(cropN, cropN, cropN),  # .transpose([1, 0, 2]),
                        level=level,
                        spacing=(
                            (x_max - x_min) / (cropN - 1),
                            (y_max - y_min) / (cropN - 1),
                            (z_max - z_min) / (cropN - 1),
                        ),
                        mask=current_mask,
                    )
                    verts = verts + np.array([x_min, y_min, z_min])

                    meshcrop = trimesh.Trimesh(verts, faces, normals)
                    meshes.append(meshcrop)

    combined = trimesh.util.concatenate(meshes)
    combined = clean_extracted_mesh(combined)

    # inverse contraction and clipping the points range
    if inv_contraction is not None:
        combined.vertices = inv_contraction(torch.from_numpy(combined.vertices)).numpy()
        combined.vertices = np.clip(combined.vertices, -max_range, max_range)

    if return_mesh:
        return combined
    else:
        filename = str(output_path)
        filename_simplify = str(output_path).replace(".ply", "-simplify.ply")

        combined.export(filename)
        if simplify_mesh:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(filename)

            print("simply mesh")
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=2000000)
            ms.save_current_mesh(filename_simplify, save_face_color=False)

@torch.no_grad()
def get_falloff(field, window_width):
    """
    Eq. 5 in the paper
    """
    inv_window_width = 1.0 / window_width
    return 0.5 * (1.0 + torch.cos(torch.pi * torch.clamp(field * inv_window_width, -1.0, 1.0)))

@torch.no_grad()
def get_field_gradient(field, spacing=(1.0, 1.0, 1.0)):
    """Calculate numerical gradient of a 3D field using central differences.
    
    Args:
        field: Tensor of shape (X, Y, Z) ordered in XYZ coordinates
        spacing: Grid spacing in (x, y, z)
    
    Returns:
        Gradient tensor of shape (3, X, Y, Z) where the first dimension 
        contains (dx, dy, dz)
    """
    dx, dy, dz = spacing
    
    # Central differences for interior points
    grad_x = (field[2:, :, :] - field[:-2, :, :]) / (2 * dx)
    grad_y = (field[:, 2:, :] - field[:, :-2, :]) / (2 * dy)
    grad_z = (field[:, :, 2:] - field[:, :, :-2]) / (2 * dz)
    
    # Forward/backward differences for boundaries
    grad_x_left = (field[1:2, :, :] - field[0:1, :, :]) / dx
    grad_x_right = (field[-1:, :, :] - field[-2:-1, :, :]) / dx
    grad_y_top = (field[:, 1:2, :] - field[:, 0:1, :]) / dy
    grad_y_bottom = (field[:, -1:, :] - field[:, -2:-1, :]) / dy
    grad_z_front = (field[:, :, 1:2] - field[:, :, 0:1]) / dz
    grad_z_back = (field[:, :, -1:] - field[:, :, -2:-1]) / dz
    
    # Combine central differences with boundary differences
    grad_x = torch.cat([grad_x_left, grad_x, grad_x_right], dim=0)
    grad_y = torch.cat([grad_y_top, grad_y, grad_y_bottom], dim=1)
    grad_z = torch.cat([grad_z_front, grad_z, grad_z_back], dim=2)
    
    # Stack gradients
    gradient = torch.stack([grad_x, grad_y, grad_z], dim=-1)
    
    return gradient

@torch.no_grad()
def get_alpha_from_field(field_grid, inv_s_grid, half_spacing):
    """
    Procedure 2 in the paper
    """
    f_offset_b_div_s, f_offset_f_div_s = (field_grid - half_spacing) * inv_s_grid, (field_grid + half_spacing) * inv_s_grid
    sigmoid_b, sigmoid_f = torch.sigmoid(f_offset_b_div_s), torch.sigmoid(f_offset_f_div_s)
    # alpha = (sigmoid_b - sigmoid_f) / (sigmoid_b + 1e-6)
    alpha = (sigmoid_f - sigmoid_b) / (sigmoid_f + 1e-6)
    return alpha
    
@torch.no_grad()
def get_velocity_d(alpha_grid, sigma_min, beta_d):
    """
    Eq. 6 in the paper
    """
    return torch.where(alpha_grid > sigma_min, beta_d * alpha_grid, torch.zeros_like(alpha_grid))

@torch.no_grad()
def smooth_scalar_grid(grid: torch.Tensor, n_iter: int) -> torch.Tensor:
    """Lightweight replicate-padded 3x3x3 smoothing for shell velocity fields."""
    n_iter = max(int(n_iter), 0)
    for _ in range(n_iter):
        grid_5d = grid[None, None, ...]
        grid = F.avg_pool3d(F.pad(grid_5d, (1, 1, 1, 1, 1, 1), mode="replicate"), 3, stride=1)[0, 0]
    return grid


@torch.no_grad()
def get_velocity_e(alpha_grid, v_max, beta_e, mode: str = "inverse_alpha", excess_floor: float = 0.0):
    """
    Eq. 7 in the paper
    """
    alpha_safe = torch.clamp(alpha_grid, min=1e-6)
    if mode == "inverse_alpha":
        speed = beta_e / alpha_safe
    elif mode == "excess_inverse_alpha":
        # Keep alpha~=1 sharp surfaces nearly fixed while still eroding low-alpha, large-kernel regions.
        speed = beta_e * torch.clamp(1.0 / alpha_safe - 1.0, min=0.0)
    elif mode == "baseline_excess_inverse_alpha":
        speed = beta_e * (float(excess_floor) + torch.clamp(1.0 / alpha_safe - 1.0, min=0.0))
    else:
        raise ValueError(f"Unknown erosion velocity mode: {mode}")
    return -torch.minimum(speed, torch.tensor(v_max, device=alpha_grid.device))

@torch.no_grad()
def get_mean_curvature(grad_normalized, spacing):
    """Calculate mean curvature as divergence of normalized gradient.
    
    Args:
        grad_normalized: Normalized gradient field of shape (X, Y, Z, 3)
        spacing: Grid spacing in (x, y, z)
    
    Returns:
        Mean curvature of shape (X, Y, Z)
    """
    dx, dy, dz = spacing
    
    # Extract normalized gradient components
    nx, ny, nz = grad_normalized[..., 0], grad_normalized[..., 1], grad_normalized[..., 2]
    
    # Calculate diagonal terms of the Hessian (∂nx/∂x, ∂ny/∂y, ∂nz/∂z)
    # Central differences for interior points
    dnx_dx = (nx[2:, :, :] - nx[:-2, :, :]) / (2 * dx)
    dny_dy = (ny[:, 2:, :] - ny[:, :-2, :]) / (2 * dy)
    dnz_dz = (nz[:, :, 2:] - nz[:, :, :-2]) / (2 * dz)
    
    # Forward/backward differences for boundaries
    dnx_dx_left = (nx[1:2, :, :] - nx[0:1, :, :]) / dx
    dnx_dx_right = (nx[-1:, :, :] - nx[-2:-1, :, :]) / dx
    dny_dy_top = (ny[:, 1:2, :] - ny[:, 0:1, :]) / dy
    dny_dy_bottom = (ny[:, -1:, :] - ny[:, -2:-1, :]) / dy
    dnz_dz_front = (nz[:, :, 1:2] - nz[:, :, 0:1]) / dz
    dnz_dz_back = (nz[:, :, -1:] - nz[:, :, -2:-1]) / dz
    
    # Combine central differences with boundary differences
    dnx_dx = torch.cat([dnx_dx_left, dnx_dx, dnx_dx_right], dim=0)
    dny_dy = torch.cat([dny_dy_top, dny_dy, dny_dy_bottom], dim=1)
    dnz_dz = torch.cat([dnz_dz_front, dnz_dz, dnz_dz_back], dim=2)
    
    # Mean curvature is the divergence of normalized gradient
    mean_curvature = dnx_dx + dny_dy + dnz_dz
    
    return mean_curvature

@torch.no_grad()
def evolve_field(
    field_grid,
    grad_grid,
    velocity,
    spacing,
    time_step,
    n_iter,
    window_width,
    weight_curv,
    recompute_grad_each_iter: bool = False,
    curvature_motion_ratio: Optional[float] = None,
):
    """
    Procedure 1 in the paper
    """
    for _ in tqdm(range(n_iter), desc="Evolving field"):
        if recompute_grad_each_iter:
            grad_grid = get_field_gradient(field_grid, spacing)
        grad_norm = torch.norm(grad_grid, dim=-1)
        motion_term = -velocity * grad_norm
        # second order gradient of normalized gradient
        grad_normalized = grad_grid / (grad_norm[..., None] + 1e-6)
        # Calculate mean curvature
        curvature = get_mean_curvature(grad_normalized, spacing)
        curvature_term = -weight_curv * grad_norm * curvature
        if curvature_motion_ratio is not None:
            limit = float(curvature_motion_ratio) * torch.abs(motion_term)
            curvature_term = torch.clamp(curvature_term, min=-limit, max=limit)
        falloff = get_falloff(field_grid, window_width)
        field_grid = field_grid + time_step * (motion_term + curvature_term) * falloff

    return field_grid

@torch.no_grad()
def clean_extracted_mesh(
    mesh: trimesh.Trimesh,
    *,
    merge_digits_vertex: int = 6,
    min_component_faces: int = 500,
    min_component_fraction: float = 0.01,
) -> trimesh.Trimesh:
    """Apply conservative cleanup to extracted meshes.

    Goals:
    - remove degenerate / duplicate / unreferenced geometry
    - merge nearly identical vertices
    - drop tiny disconnected floaters while keeping main components
    """
    mesh = mesh.copy()

    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        mesh.merge_vertices(digits_vertex=merge_digits_vertex)
    except Exception:
        pass

    try:
        mesh.remove_duplicate_faces()
    except Exception:
        pass

    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass

    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        components = list(mesh.split(only_watertight=False))
    except Exception:
        components = [mesh]

    if len(components) <= 1:
        return mesh

    largest_faces = max(len(comp.faces) for comp in components)
    face_threshold = max(min_component_faces, int(largest_faces * min_component_fraction))
    kept_components = [comp for comp in components if len(comp.faces) >= face_threshold]
    if len(kept_components) == 0:
        kept_components = [max(components, key=lambda comp: len(comp.faces))]

    cleaned = trimesh.util.concatenate(kept_components)

    try:
        cleaned.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        cleaned.merge_vertices(digits_vertex=merge_digits_vertex)
    except Exception:
        pass

    return cleaned


@torch.no_grad()
def _print_tensor_stats(name: str, values: torch.Tensor) -> None:
    values = values.detach().cpu()
    flat = values.reshape(-1).float()
    finite = torch.isfinite(flat)
    finite_count = int(finite.sum().item())
    total_count = int(flat.numel())
    if finite_count == 0:
        print(f"[shell-stats] {name}: no finite values")
        return

    flat = flat[finite]
    quantile_input = flat
    max_quantile_samples = 2_000_000
    if flat.numel() > max_quantile_samples:
        sample_idx = torch.arange(max_quantile_samples, dtype=torch.int64)
        sample_idx = (sample_idx * flat.numel()) // max_quantile_samples
        sample_idx = torch.clamp(sample_idx, max=flat.numel() - 1)
        quantile_input = flat[sample_idx]
    quantiles = torch.quantile(
        quantile_input,
        torch.tensor([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]),
    )
    print(
        "[shell-stats] "
        f"{name}: "
        f"shape={tuple(values.shape)} "
        f"finite={finite_count}/{total_count} "
        f"min={quantiles[0].item():.6g} "
        f"q01={quantiles[1].item():.6g} "
        f"q05={quantiles[2].item():.6g} "
        f"median={quantiles[3].item():.6g} "
        f"q95={quantiles[4].item():.6g} "
        f"q99={quantiles[5].item():.6g} "
        f"max={quantiles[6].item():.6g} "
        f"mean={flat.mean().item():.6g} "
        f"std={flat.std(unbiased=False).item():.6g}"
    )


@torch.no_grad()
def _print_levelset_extraction_stats(
    name: str,
    values,
    level: float,
) -> None:
    values_np = np.asarray(values.detach().cpu(), dtype=np.float32)
    flat = values_np.reshape(-1)
    finite = np.isfinite(flat)
    if not np.any(finite):
        print(f"[mesh-stats] {name}: no finite values")
        return

    flat = flat[finite]
    quantiles = np.quantile(flat, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    min_val = float(quantiles[0])
    max_val = float(quantiles[-1])
    crosses_level = min_val <= level <= max_val
    nearest_delta = min(float(np.min(np.abs(flat - level))), abs(min_val - level), abs(max_val - level))
    neg_frac = float(np.mean(flat < level))
    pos_frac = float(np.mean(flat > level))
    print(
        "[mesh-stats] "
        f"{name}: "
        f"level={level:.6g} "
        f"crosses={crosses_level} "
        f"min={min_val:.6g} "
        f"q01={float(quantiles[1]):.6g} "
        f"q05={float(quantiles[2]):.6g} "
        f"median={float(quantiles[3]):.6g} "
        f"q95={float(quantiles[4]):.6g} "
        f"q99={float(quantiles[5]):.6g} "
        f"max={max_val:.6g} "
        f"nearest_abs_delta={nearest_delta:.6g} "
        f"below_level_frac={neg_frac:.6f} "
        f"above_level_frac={pos_frac:.6f}"
    )


@torch.no_grad()
def _print_shell_extraction_stats(
    *,
    field_grid: torch.Tensor,
    inv_s_grid: torch.Tensor,
    grad_field: torch.Tensor,
    alpha_grid: torch.Tensor,
    velocity_d: torch.Tensor,
    velocity_e: torch.Tensor,
    field_grid_d: torch.Tensor,
    field_grid_e: torch.Tensor,
    sigma_min: float,
    time_step_d: float,
    time_step_e: float,
    n_iter: int,
    weight_curv_d: float,
    weight_curv_e: float,
    window_width_d: float,
    window_width_e: float,
    beta_d: float,
    beta_e: float,
    v_max: float,
    spacing: tuple,
    recompute_grad_each_iter: bool,
    erosion_velocity_mode: str,
    erosion_excess_floor: float,
    smooth_velocity_d_iters: int,
    smooth_velocity_e_iters: int,
    curvature_motion_ratio_d: Optional[float],
    curvature_motion_ratio_e: Optional[float],
) -> None:
    grad_norm = torch.linalg.norm(grad_field, dim=-1)
    delta_d = field_grid_d - field_grid
    delta_e = field_grid_e - field_grid
    print("[shell-stats] -------- shell extraction summary --------")
    print(
        "[shell-stats] "
        f"spacing={spacing} "
        f"time_step_d={time_step_d} time_step_e={time_step_e} n_iter={n_iter} "
        f"sigma_min={sigma_min} beta_d={beta_d} beta_e={beta_e} v_max={v_max} "
        f"weight_curv_d={weight_curv_d} weight_curv_e={weight_curv_e} "
        f"window_width_d={window_width_d} window_width_e={window_width_e} "
        f"recompute_grad_each_iter={recompute_grad_each_iter} "
        f"erosion_velocity_mode={erosion_velocity_mode} "
        f"erosion_excess_floor={erosion_excess_floor} "
        f"smooth_velocity_d_iters={smooth_velocity_d_iters} smooth_velocity_e_iters={smooth_velocity_e_iters} "
        f"curvature_motion_ratio_d={curvature_motion_ratio_d} curvature_motion_ratio_e={curvature_motion_ratio_e}"
    )
    _print_tensor_stats("field_grid", field_grid)
    _print_tensor_stats("inv_s_grid", inv_s_grid)
    _print_tensor_stats("grad_norm", grad_norm)
    _print_tensor_stats("alpha_grid", alpha_grid)
    _print_tensor_stats("velocity_d", velocity_d)
    _print_tensor_stats("velocity_e", velocity_e)
    _print_tensor_stats("delta_d", delta_d)
    _print_tensor_stats("delta_e", delta_e)
    surface_mask = torch.abs(field_grid) <= max(float(spacing[0]) * 2.0, 1e-6)
    erosion_window_mask = torch.abs(field_grid) <= float(window_width_e)
    dilation_window_mask = torch.abs(field_grid) <= float(window_width_d)
    if bool(torch.any(surface_mask).item()):
        _print_tensor_stats("alpha_grid_near_surface", alpha_grid[surface_mask])
        _print_tensor_stats("velocity_d_near_surface", velocity_d[surface_mask])
        _print_tensor_stats("velocity_e_near_surface", velocity_e[surface_mask])
        _print_tensor_stats("delta_d_near_surface", delta_d[surface_mask])
        _print_tensor_stats("delta_e_near_surface", delta_e[surface_mask])
    if bool(torch.any(dilation_window_mask).item()):
        _print_tensor_stats("delta_d_dilation_window", delta_d[dilation_window_mask])
    if bool(torch.any(erosion_window_mask).item()):
        _print_tensor_stats("delta_e_erosion_window", delta_e[erosion_window_mask])
    print(
        "[shell-stats] "
        f"alpha_gt_sigma_min={float((alpha_grid > sigma_min).float().mean().item()):.6f} "
        f"velocity_d_nonzero={float((velocity_d > 0).float().mean().item()):.6f} "
        f"velocity_e_at_vmax={float((velocity_e <= (-v_max + 1e-6)).float().mean().item()):.6f} "
        f"dilation_changed={float((delta_d != 0).float().mean().item()):.6f} "
        f"erosion_changed={float((delta_e != 0).float().mean().item()):.6f}"
    )
    print("[shell-stats] ----------------------------------------")


@torch.no_grad()
def get_mesh(field_grid, level, spacing, bounding_box_min):
    # Extract mesh using marching cubes
    verts, faces, normals, _ = measure.marching_cubes(
        volume=field_grid,
        level=level,
        spacing=spacing,
    )
    
    # Transform vertices to world space
    verts = verts + np.array(bounding_box_min)
    
    # Create and clean mesh
    mesh = trimesh.Trimesh(verts, faces, normals)
    mesh = clean_extracted_mesh(mesh)
    return mesh

@torch.no_grad()
def _evaluate_regular_grid_field(
    field_fn,
    *,
    bounding_box_min=(-1.0, -1.0, -1.0),
    bounding_box_max=(1.0, 1.0, 1.0),
    resolution=512,
    value_dims=1,
    device="cuda",
    slab_depth=8,
    batch_points=200000,
):
    """Evaluate a field on a regular 3D grid without materializing all grid points at once."""
    device = torch.device(device)
    x = torch.linspace(bounding_box_min[0], bounding_box_max[0], resolution, dtype=torch.float32, device=device)
    y = torch.linspace(bounding_box_min[1], bounding_box_max[1], resolution, dtype=torch.float32, device=device)
    z = torch.linspace(bounding_box_min[2], bounding_box_max[2], resolution, dtype=torch.float32, device=device)

    values = torch.empty((resolution, resolution, resolution, value_dims), dtype=torch.float32, device=device)
    slab_depth = max(int(slab_depth), 1)
    batch_points = max(int(batch_points), 1)

    for z_start in tqdm(range(0, resolution, slab_depth), desc="Evaluating grid"):
        z_end = min(z_start + slab_depth, resolution)
        z_slice = z[z_start:z_end]
        xx, yy, zz = torch.meshgrid(x, y, z_slice, indexing="ij")
        slab_points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

        slab_values = torch.empty((slab_points.shape[0], value_dims), dtype=torch.float32, device=device)
        write_head = 0
        for pts in torch.split(slab_points, batch_points, dim=0):
            chunk_values = field_fn(pts)
            slab_values[write_head : write_head + pts.shape[0]] = chunk_values[:, :value_dims].to(torch.float32)
            write_head += pts.shape[0]

        values[:, :, z_start:z_end, :] = slab_values.view(resolution, resolution, z_end - z_start, value_dims)

    return values


@torch.no_grad()
def extract_mesh_shell(
    field_fn,
    bounding_box_min=(-1.0, -1.0, -1.0),
    bounding_box_max=(1.0, 1.0, 1.0),
    level: float = 0.0,
    output_path: Path = Path("mesh.ply"),
    return_mesh=False,
    simplify_mesh=False,
    resolution=512,
    n_iter=50,
    time_step: float = 0.01, # 0.1
    time_step_d: Optional[float] = None,
    time_step_e: Optional[float] = None,
    sigma_min=0.01,
    beta_d: float = 1.0,
    beta_e: float = 0.001,
    v_max: float = 100.0,
    weight_curv_d: float = 0.0001, # 0.01
    window_width_d: float = 0.1,
    weight_curv_e: float = 0.0,
    window_width_e: float = 0.05,
    kernel_size_transform_mode="exp",
    kernel_size_transform_exp_factor=10.0,
    recompute_grad_each_iter: bool = False,
    erosion_velocity_mode: str = "inverse_alpha",
    erosion_excess_floor: float = 0.0,
    smooth_velocity_d_iters: int = 0,
    smooth_velocity_e_iters: int = 0,
    curvature_motion_ratio_d: Optional[float] = None,
    curvature_motion_ratio_e: Optional[float] = None,
    debug_stats=False,
    device="cuda",
    slab_depth=8,
    batch_points=200000,
):
    """Extract mesh from a field function using marching cubes.
    
    Args:
        field_fn: Callable that takes (N, 3) points and returns (N,) field values
        bounding_box_min: Minimum coordinates (x,y,z) of the bounding box
        bounding_box_max: Maximum coordinates (x,y,z) of the bounding box
        level: Isosurface level to extract
        output_path: Path to save the mesh
        return_mesh: If True, returns the trimesh object
        simplify_mesh: If True, simplifies the output mesh
    """
    assert resolution == 512, "Resolution must be 512 for now, chunking is not supported"

    raw_grid = _evaluate_regular_grid_field(
        field_fn,
        bounding_box_min=bounding_box_min,
        bounding_box_max=bounding_box_max,
        resolution=resolution,
        value_dims=2,
        device=device,
        slab_depth=slab_depth,
        batch_points=batch_points,
    )
    field_grid = raw_grid[..., 0]
    inv_s_grid = transform_inv_s(
        raw_grid[..., 1],
        mode=kernel_size_transform_mode,
        exp_factor=kernel_size_transform_exp_factor,
    )
    if debug_stats:
        _print_levelset_extraction_stats("base_field_grid", field_grid, level)

    # Skip if no surface found
    if field_grid.min() > level or field_grid.max() < level:
        print("No surface found for base mesh extraction")
        if not debug_stats:
            _print_levelset_extraction_stats("base_field_grid", field_grid, level)
        return None if return_mesh else None
    
    spacing = (
        (bounding_box_max[0] - bounding_box_min[0]) / (resolution - 1),
        (bounding_box_max[1] - bounding_box_min[1]) / (resolution - 1),
        (bounding_box_max[2] - bounding_box_min[2]) / (resolution - 1),
    ) # shape: (3,)
    spacing_tensor = torch.tensor(spacing, device=field_grid.device)[None, None, None, :]
    # Procedure 2
    grad_field = get_field_gradient(field_grid, spacing)
    assert spacing[0] == spacing[1] == spacing[2], "Spacing must be the same in all dimensions"
    alpha_grid = get_alpha_from_field(field_grid, inv_s_grid, spacing_tensor[..., 0] * 0.5)
    # dilation
    if time_step_d is None:
        time_step_d = time_step
    if time_step_e is None:
        time_step_e = time_step

    velocity_d = get_velocity_d(alpha_grid, sigma_min, beta_d)
    velocity_d = smooth_scalar_grid(velocity_d, smooth_velocity_d_iters)
    field_grid_d = evolve_field(
        field_grid,
        grad_field,
        velocity_d,
        spacing,
        float(time_step_d),
        n_iter,
        window_width_d,
        weight_curv_d,
        recompute_grad_each_iter=recompute_grad_each_iter,
        curvature_motion_ratio=curvature_motion_ratio_d,
    )
    field_grid_d = torch.minimum(field_grid, field_grid_d)
    # erosion
    velocity_e = get_velocity_e(
        alpha_grid,
        v_max,
        beta_e,
        mode=erosion_velocity_mode,
        excess_floor=erosion_excess_floor,
    )
    velocity_e = smooth_scalar_grid(velocity_e, smooth_velocity_e_iters)
    field_grid_e = evolve_field(
        field_grid,
        grad_field,
        velocity_e,
        spacing,
        float(time_step_e),
        n_iter,
        window_width_e,
        weight_curv_e,
        recompute_grad_each_iter=recompute_grad_each_iter,
        curvature_motion_ratio=curvature_motion_ratio_e,
    )
    field_grid_e = torch.maximum(field_grid, field_grid_e)

    if debug_stats:
        _print_shell_extraction_stats(
            field_grid=field_grid,
            inv_s_grid=inv_s_grid,
            grad_field=grad_field,
            alpha_grid=alpha_grid,
            velocity_d=velocity_d,
            velocity_e=velocity_e,
            field_grid_d=field_grid_d,
            field_grid_e=field_grid_e,
            sigma_min=sigma_min,
            time_step_d=float(time_step_d),
            time_step_e=float(time_step_e),
            n_iter=n_iter,
            weight_curv_d=weight_curv_d,
            weight_curv_e=weight_curv_e,
            window_width_d=window_width_d,
            window_width_e=window_width_e,
            beta_d=beta_d,
            beta_e=beta_e,
            v_max=v_max,
            spacing=spacing,
            recompute_grad_each_iter=recompute_grad_each_iter,
            erosion_velocity_mode=erosion_velocity_mode,
            erosion_excess_floor=erosion_excess_floor,
            smooth_velocity_d_iters=smooth_velocity_d_iters,
            smooth_velocity_e_iters=smooth_velocity_e_iters,
            curvature_motion_ratio_d=curvature_motion_ratio_d,
            curvature_motion_ratio_e=curvature_motion_ratio_e,
        )

    print("Extracting meshes...")


    if return_mesh:
        mesh = get_mesh(field_grid.cpu().numpy(), level, spacing, bounding_box_min)
        mesh_d = get_mesh(field_grid_d.cpu().numpy(), level, spacing, bounding_box_min)
        mesh_e = get_mesh(field_grid_e.cpu().numpy(), level, spacing, bounding_box_min)        
        return mesh, mesh_d, mesh_e
    else:
        # Save mesh
        output_path.parent.mkdir(parents=True, exist_ok=True)
        filename = str(output_path)
        filename_d = str(output_path).replace(".ply", "-dilation.ply")
        filename_e = str(output_path).replace(".ply", "-erosion.ply")
        filename_simplify = str(output_path).replace(".ply", "-simplify.ply")
        mesh = get_mesh(field_grid.cpu().numpy(), level, spacing, bounding_box_min)
        mesh.export(filename)
        mesh_d = get_mesh(field_grid_d.cpu().numpy(), level, spacing, bounding_box_min)
        mesh_d.export(filename_d)
        mesh_e = get_mesh(field_grid_e.cpu().numpy(), level, spacing, bounding_box_min)
        mesh_e.export(filename_e)

        if simplify_mesh:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(filename)
            print("Simplifying mesh...")
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=2000000)
            ms.save_current_mesh(filename_simplify, save_face_color=False)


@torch.no_grad()
def extract_mesh_simple(
    field_fn,
    bounding_box_min=(-1.0, -1.0, -1.0),
    bounding_box_max=(1.0, 1.0, 1.0),
    level: float = 0.0,
    output_path: Path = Path("mesh.ply"),
    return_mesh=False,
    simplify_mesh=False,
    resolution=512,
    device="cuda",
    slab_depth=8,
    batch_points=200000,
):
    """Extract mesh from a field function using marching cubes.
    
    Args:
        field_fn: Callable that takes (N, 3) points and returns (N,) field values
        bounding_box_min: Minimum coordinates (x,y,z) of the bounding box
        bounding_box_max: Maximum coordinates (x,y,z) of the bounding box
        level: Isosurface level to extract
        output_path: Path to save the mesh
        return_mesh: If True, returns the trimesh object
        simplify_mesh: If True, simplifies the output mesh
    """
    assert resolution == 512, "Resolution must be 512 for now, chunking is not supported"

    field_grid = _evaluate_regular_grid_field(
        field_fn,
        bounding_box_min=bounding_box_min,
        bounding_box_max=bounding_box_max,
        resolution=resolution,
        value_dims=1,
        device=device,
        slab_depth=slab_depth,
        batch_points=batch_points,
    )[..., 0].cpu().numpy()
    _print_levelset_extraction_stats("simple_field_grid", field_grid, level)

    # Skip if no surface found
    if np.min(field_grid) > level or np.max(field_grid) < level:
        print("No surface found for simple mesh extraction")
        return None if return_mesh else None

    # Extract mesh using marching cubes
    verts, faces, normals, _ = measure.marching_cubes(
        volume=field_grid,
        level=level,
        spacing=(
            (bounding_box_max[0] - bounding_box_min[0]) / (resolution - 1),
            (bounding_box_max[1] - bounding_box_min[1]) / (resolution - 1),
            (bounding_box_max[2] - bounding_box_min[2]) / (resolution - 1),
        ),
    )
    
    # Transform vertices to world space
    verts = verts + np.array(bounding_box_min)
    
    # Create and clean mesh
    mesh = trimesh.Trimesh(verts, faces, normals)
    mesh = clean_extracted_mesh(mesh)

    if return_mesh:
        return mesh
    else:
        # Save mesh
        output_path.parent.mkdir(parents=True, exist_ok=True)
        filename = str(output_path)
        filename_simplify = str(output_path).replace(".ply", "-simplify.ply")
        mesh.export(filename)
        
        if simplify_mesh:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(filename)
            print("Simplifying mesh...")
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=2000000)
            ms.save_current_mesh(filename_simplify, save_face_color=False)
