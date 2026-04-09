"""
Hyperparameters for adversarial attack and defense experiments.
"""


class CFG:
    image_size = 256
    seed = 42

    # Attack parameters
    search_iters = 50
    population_size = 10

    # Defense 1: Randomized Preprocessing with TTA
    tta_K = 5
    tta_resize_range = (0.9, 1.1)
    tta_blur_range = (0.3, 1.5)

    # Defense 2: Deterministic Input Denoising
    denoise_sigma = 1.0
    denoise_median = 3

    # Defense 4: Stochastic Ensemble
    ensemble_K = 5

    # Statistical analysis
    n_bootstrap = 1000
