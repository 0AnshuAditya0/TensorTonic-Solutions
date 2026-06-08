import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """

    total_returns = np.zeros(n_states)
    state_counts = np.zeros(n_states)
    
    for episode in episodes:
        
        first_visits = {}
        for idx, (state, reward) in enumerate(episode):
            if state not in first_visits:
                first_visits[state] = idx
                
        for state, first_idx in first_visits.items():
            g_return = 0
            discount = 1
            
            for idx in range(first_idx, len(episode)):
                g_return += discount * episode[idx][1]
                discount *= gamma
                
            total_returns[state] += g_return
            state_counts[state] += 1

    v_values = np.zeros(n_states)
    for state in range(n_states):
        if state_counts[state] > 0:
            v_values[state] = total_returns[state] / state_counts[state]
            
    return v_values
