import torch

def apply_semantic_edit(z, delta_c, mask, alpha=1.0):
    """
    Applies the latent intervention: z_edit = z + alpha * (mask * delta_c)
    This implementation follows the Projection-Before-Attention framework.
    """
    # We use unsqueeze(-1) to make sure the mask [Patches] 
    # matches the latent dimension [Patches, Dimension]
    return z + alpha * (mask.unsqueeze(-1) * delta_c)
