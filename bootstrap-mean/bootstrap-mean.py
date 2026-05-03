import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    x = np.array(x)
    resamples = rng.choice(x, size=(n_bootstrap, len(x)), replace=True)
    
    boot_means = np.mean(resamples, axis=1)

    alpha = 1 - ci
    lower = np.percentile(boot_means, (alpha / 2) * 100)
    upper = np.percentile(boot_means, (1 - alpha / 2) * 100)
    
    return boot_means, lower, upper
