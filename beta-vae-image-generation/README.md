# Beta-VAE Controllable Image Generation

Implementation of Beta-Variational Autoencoder for controllable facial image generation with disentangled latent representations.

## Project Overview

This project explores the fundamental trade-off in generative modeling between reconstruction fidelity and latent space interpretability. By implementing Beta-VAE with systematic $\beta$-parameter optimization, we achieve controllable image generation while maintaining high visual quality.

## Technical Approach

### Core Architecture
- **Encoder**: Convolutional layers reducing 64×64×3 images to 128-dimensional latent vectors
- **Decoder**: Transposed convolutional layers reconstructing images from latent codes
- **Loss Function**: `Total Loss = Reconstruction Loss + $\beta$ × KL Divergence Loss`

### Key Innovation: $\beta$-Parameter Optimization
- **$\beta$ = 1**: Standard VAE (balanced reconstruction/regularization)
- **$\beta$ = 4**: **Optimal configuration** - 97% reconstruction quality with enhanced controllability
- **$\beta$ = 10**: Maximum disentanglement with acceptable quality trade-off

## Results & Insights

### Quantitative Performance
- **Optimal $\beta$ = 4**: Best balance between reconstruction (120.53 loss) and disentanglement (289.25 KLD)
- **Training Efficiency**: Stable convergence across 12 epochs on CelebA subset
- **Latent Space Quality**: Achieved meaningful feature traversals and smooth interpolations

### Key Findings
1. **Reconstruction-Disentanglement Trade-off**: Contrary to common belief, meaningful disentanglement achievable with minimal reconstruction degradation (2-7% loss)
2. **$\beta$-Parameter Sensitivity**: Systematic evaluation reveals $\beta$=4 as optimal for facial image generation
3. **Practical Controllability**: Higher $\beta$ values enable more predictable feature manipulation

## Technical Implementation

### Technologies Used
- **PyTorch**: Deep learning framework and model implementation
- **CelebA Dataset**: 5,000 high-quality facial images for training/validation
- **Apple Silicon (MPS)**: Optimized for M-series chip acceleration
- **Custom Evaluation Framework**: Multi-metric assessment pipeline

### Engineering Highlights
- Device-agnostic processing with automatic MPS/CUDA/CPU detection
- Memory-optimized batch processing for resource-constrained environments
- Comprehensive visualization tools for latent space analysis
- Reproducible research practices with systematic hyperparameter logging

## Applications & Impact

### Product Implications
- **Interactive Face Editing**: Users can modify specific attributes (age, expression, hair)
- **Content Generation**: Controllable synthetic data for training/augmentation
- **Creative Tools**: Professional-grade face manipulation for media production

### Technical Contributions
- Demonstrated optimal $\beta$-parameter selection methodology
- Validated reconstruction-disentanglement trade-off characteristics
- Established evaluation framework for generative model controllability

## Performance Metrics

- **Reconstruction Quality**: MSE loss tracking across $\beta$ configurations
- **Disentanglement Measure**: KL divergence analysis for latent structure
- **Visual Quality Assessment**: Systematic evaluation through latent traversals
- **Computational Efficiency**: Training time optimization for practical deployment