import torch

def compute_locality_metrics(z_original, z_edited, mask):
    """
    Calculates LCS (Local Change Strength) and NTCS (Non-Target Change Strength).
    LCS: Change in the masked (target) region.
    NTCS: Change in the unmasked (non-target) region.
    """
    diff = torch.norm(z_edited - z_original, dim=-1)
    
    # Target region (where mask is 1)
    lcs = diff[mask == 1].mean() if (mask == 1).any() else 0.0
    
    # Non-target region (where mask is 0)
    ntcs = diff[mask == 0].mean() if (mask == 0).any() else 0.0
    
    return lcs, ntcs
