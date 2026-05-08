import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    values = np.array(values)
    transitions = np.array(transitions)
    rewards = np.array(rewards)
    
    num_states = len(values)
    new_values = np.zeros(num_states)
    
    for s in range(num_states):
        q_values = []
        for a in range(len(transitions[s])):
            expected_future_value = np.dot(transitions[s][a], values)
            q_s_a = rewards[s][a] + gamma * expected_future_value
            q_values.append(q_s_a)
        
        new_values[s] = max(q_values)
        
    return list(new_values)