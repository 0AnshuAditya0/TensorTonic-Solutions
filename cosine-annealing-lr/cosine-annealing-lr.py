def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    """
    Compute the learning rate using cosine annealing.
    """
    if current_step >= total_steps:
        return min_lr

    cosine_factor = math.cos(math.pi * current_step / total_steps)

    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cosine_factor)
    
    return lr