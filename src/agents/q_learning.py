import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount=0.95, epsilon=0.1):
        self.action_space = action_space
        self.lr = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
    
    def _state_to_key(self, state):
        # Convert state to hashable key (simplified)
        return tuple(np.round(state, 2))
    
    def select_action(self, state):
        state_key = self._state_to_key(state)
        
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state):
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        current_q = self.q_table[state_key][action]
        next_max_q = np.max(self.q_table[next_state_key])
        
        self.q_table[state_key][action] = current_q + self.lr * (
            reward + self.discount * next_max_q - current_q
        )