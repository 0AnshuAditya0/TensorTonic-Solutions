import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    """
    Returns: updated value function V_new
    """
    V_new = np.array(V, dtype=float)
    td_target = r + gamma * V_new[s_next]
    td_error = td_target - V_new[s]

    V_new[s] += alpha * td_error
    
    return V_new.tolist()