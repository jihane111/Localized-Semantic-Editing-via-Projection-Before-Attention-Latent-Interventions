import torch
import torch.nn as nn

class FAE(nn.Module):
    """
    Feature Auto-Encoder (FAE)
    This model projects features into a latent space where we can edit them.
    """
    def __init__(self, input_dim=1024, latent_dim=512):
        super(FAE, self).__init__()
        # Linear projection: This is where the 'Projection' happens!
        self.projection = nn.Linear(input_dim, latent_dim)
        self.attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=8)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x, mask=None, delta_c=None, alpha=0.0):
        # 1. Project to Latent Space
        z = self.projection(x)
        
        # 2. INNOVATION: Apply edit BEFORE Attention
        if mask is not None and delta_c is not None:
            z = z + alpha * (mask.unsqueeze(-1) * delta_c)
            
        # 3. Apply Attention (Contextualization)
        z_attn, _ = self.attention(z, z, z)
        
        # 4. Decode back
        return self.decoder(z_attn)
