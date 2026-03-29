# Localized Semantic Editing via Projection-Before-Attention

Official implementation of the **Projection-Before-Attention** framework for fine-tuning-free latent interventions. This research addresses the problem of "semantic leakage" in Transformer-based generative models.

## 📖 Abstract
Traditional latent editing often causes unintended changes in non-target regions due to the global nature of Multi-Head Attention. This project introduces a reordered Feature Auto-Encoder (FAE) architecture where additive edits are applied **after linear projection but before attention**. This ensures that semantic modifications are spatially confined to user-defined masks.

## ✨ Key Contributions
* **Projection-Before-Attention:** A novel architectural shift ensuring patch-level isolation.
* **Zero Fine-Tuning:** High-fidelity attribute control (e.g., Smiling, Eyeglasses) performed entirely at inference time.
* **Mathematical Locality:** Provable confinement of edits within binary spatial masks ($M$).

## 🛠 Project Structure
* `src/model.py`: FAE architecture featuring the reordered forward pass.
* `src/intervention.py`: Logic for additive semantic editing ($\delta_c$).
* `src/metrics.py`: Implementation of LCS (Local Change Strength) and NTCS (Non-Target Change Strength).
* `src/utils.py`: Spatial masking and latent normalization utilities.
* `tests/`: Unit tests for Theorem 1 and Counterfactual Stress Tests.
* `Dockerfile`: Containerization for reproducible research environments.

## 📈 Training & Evaluation
The FAE is trained on the **CelebA** dataset using **Mean Squared Error (MSE)** loss to ensure a stable, reconstructible latent space. Evaluation is performed using the Locality Ratio ($LCS/NTCS$), demonstrating minimal leakage even at high intervention intensities ($\alpha$).

## 🚀 Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run locality verification: `python -m tests.test_counterfactual`
