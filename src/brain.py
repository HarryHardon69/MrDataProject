import numpy as np
from environment import Environment

class MrDataAgent:
    def __init__(self, environment):
        self.environment = environment
        self.Q_table = np.zeros((environment.num_states, environment.num_actions))
        self.gamma = 0.9
        self.epsilon = 0.1
        self.persona = Persona()

    def act(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.environment.num_actions)
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
            state = self.environment.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done = self.environment.step(action)
                self.learn(state, action, reward, next_state)
                state = next_state

    def process_input(self, user_input):
        if user_input.lower() == 'hello':
            response = "Hello! How can I assist you today?"
        elif user_input.lower() == 'status':
            response = "I am fully operational and ready to assist."
        else:
            response = "I'm sorry, I didn't understand that command."
        return response
