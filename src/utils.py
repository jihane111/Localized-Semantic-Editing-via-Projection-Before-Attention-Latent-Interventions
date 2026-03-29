import torch

def generate_patch_mask(grid_size, target_patches):
    """
    Generates a binary spatial mask for latent interventions.
    grid_size: e.g., 16 (for a 16x16 patch grid)
    target_patches: List of indices to be edited.
    """
    mask = torch.zeros(grid_size * grid_size)
    mask[target_patches] = 1.0
    return mask

def normalize_latent(z):
    """Standardizes latents before intervention."""
    return (z - z.mean(dim=0)) / (z.std(dim=0) + 1e-6)
