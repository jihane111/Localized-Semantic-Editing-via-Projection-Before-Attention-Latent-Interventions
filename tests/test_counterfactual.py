import torch
from src.intervention import apply_semantic_edit

def test_counterfactual_leakage():
    # Create a dummy latent
    z = torch.randn(16, 512)
    delta_c = torch.randn(16, 512)
    
    # Apply edit to Region A
    mask_A = torch.zeros(16)
    mask_A[0:4] = 1.0
    z_edited = apply_semantic_edit(z, delta_c, mask_A)
    
    # Check Region B (indices 8-12)
    # This proves that editing the mouth doesn't change the eyes.
    change_in_B = torch.norm(z_edited[8:12] - z[8:12])
    assert change_in_B == 0, "Counterfactual Failure: Leakage detected!"
    print("Counterfactual Stress Test: Passed")

if __name__ == "__main__":
    test_counterfactual_leakage()
