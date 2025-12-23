# UABDF: Uncertainty-Aware Balanced Deep Forest

This repository provides the official implementation of **UABDF (Uncertainty-Aware Balanced Deep Forest)**,  
a novel ensemble framework designed to improve **classification robustness**, **uncertainty modeling**,  
and **distribution alignment** in real-world **imbalanced learning** scenarios.

UABDF integrates **evidence-based uncertainty estimation**, **dynamic undersampling**,  
and a **dual-purpose Wasserstein alignment mechanism** to enhance both predictive performance  
and structural consistency across classifiers.

![UABDF Framework](imgs/UABDF_framework.jpg)


## Highlights

- **Novel Uncertainty-Aware Ensemble Framework**  
  We propose **UABDF (Uncertainty-Aware Balanced Deep Forest)**, a new ensemble framework that integrates uncertainty estimation into the learning process. The model provides adaptive and reliable solutions for class imbalance, significantly improving robustness and generalization under noisy and imbalanced scenarios.

- **Higher-Order Inference via Subjective Logic**  
  Inspired by Subjective Logic and the Beta distribution, UABDF elevates standard ensemble prediction into a form of higher-order probabilistic inference. This transforms raw classifier scores into uncertainty-aware evidence opinions, enhancing prediction credibility and decision reliability.

- **Uncertainty-Guided Dynamic Undersampling**  
  UABDF introduces an uncertainty-aware sampling strategy that performs effective undersampling without increasing computational complexity. As an extension of dynamic sampling methods, it adaptively adjusts sampling weights to balance prediction error and classifier consistency, improving robustness on highly imbalanced datasets.

- **Consensus-Based Opinion Aggregation**  
  The framework aggregates classifier opinions using evidence-guided consensus mechanisms, maintaining consistency and stability across ensemble members and reducing sensitivity to noise.

- **Wasserstein-Based Distribution Alignment**  
  UABDF aligns the predicted class distribution with the true label distribution using Wasserstein distance. This dual-purpose alignment mechanism effectively handles the trade-off between prediction consistency and classification accuracy.

## Requirements

### Main dependencies

- **Python** (>=3.7)
- **NumPy** (1.24.3)
- **pandas** (2.2.2)
- **scikit-learn** (1.3.1)
- **imbalanced-ensemble** (0.2.1)
- **ucimlrepo** (0.0.7)
- **matplotlib** (3.8.4, optional, for visualization)

> **Note:**  
> UABDF does *not* rely on deep learning frameworks (e.g., PyTorch/TensorFlow),  
> making the framework lightweight and easy to run on CPU.

## Running demo.py
To run the demo script and see UABDF in action, execute the following command in your terminal:

```bash
python demo.py
```