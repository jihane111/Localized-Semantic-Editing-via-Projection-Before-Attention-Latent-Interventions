# Localized Semantic Editing via Projection-Before-Attention

Official implementation of the **Projection-Before-Attention** framework for fine-tuning-free latent interventions.

## 🚀 The Innovation
Most Transformer-based edits suffer from "semantic leakage" because they intervene after the attention mechanism. My research proves that by intervening **after linear projection but before attention**, we can achieve:
* **Strict Spatial Locality:** Edits only affect targeted patches.
* **Zero Fine-Tuning:** No model weights are changed.
* **Linear Control:** Semantic intensity scales predictably with $\alpha$.

## 📂 Dataset
This project uses the **CelebA** dataset. To reproduce the results, ensure the dataset is structured as per the standard torchvision format or use the provided latent extraction scripts in the `notebooks/` folder.

## 📁 Project Structure
* `src/model.py`: FAE architecture with reordered forward pass.
* `src/intervention.py`: Logic for additive semantic editing and concept vectors.
* `requirements.txt`: Environment setup.

## 🧪 Theorem 1 Verification
I have implemented unit tests to verify that for any patch $p$ where the mask $M_p = 0$, the change in representation is mathematically null, ensuring 100% isolation.
