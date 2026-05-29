def gae(rewards, values, gamma, lam):
    """
    Compute Generalized Advantage Estimation.
    """
    advantages = []
    gae_accumulator = 0
    
    for t in reversed(range(len(rewards))):
        
        td_error = rewards[t] + gamma * values[t + 1] - values[t]

        gae_accumulator = td_error + gamma * lam * gae_accumulator
        
        advantages.insert(0, gae_accumulator)
        
    return advantages