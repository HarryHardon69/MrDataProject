# src/intrinsic_motivation.py
import numpy as np

class IntrinsicMotivation:
    def __init__(self):
        self.Q_table = np.zeros((100, 10))  # Example sizes
        self.gamma = 0.9
        self.epsilon = 0.1

    def act(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, 10)
        else:
            action = np.argmax(self.Q_table[state, :])
        return action

    def learn(self, state, action, reward, next_state):
        Q_value = self.Q_table[state, action]
        max_Q_value = np.max(self.Q_table[next_state, :])
        new_Q_value = reward + self.gamma * max_Q_value
        self.Q_table[state, action] = new_Q_value

    def run(self, num_episodes):
        for episode in range(num_episodes):
            state = np.random.randint(0, 100)
            done = False
            while not done:
                action = self.act(state)
                next_state = np.random.randint(0, 100)
                reward = np.random.random()
                done = np.random.random() > 0.95
                self.learn(state, action, reward, next_state)
                state = next_state
