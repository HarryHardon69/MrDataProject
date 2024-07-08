import numpy as np
from environment import Environment

class MrDataAgent:
    def __init__(self, environment):
        """
        Initializes the MrDataAgent class with the environment it will interact with.

        Args:
            environment (object): The environment the agent will interact with.
        """
        self.environment = environment  # Stores the environment reference
        self.Q_table = np.zeros((environment.num_states, environment.num_actions))  # Initializes the Q-table with zeros
        self.gamma = 0.9  # Sets the discount factor
        self.epsilon = 0.1  # Sets the exploration rate

    def act(self, state):
        """
        Selects an action using the epsilon-greedy policy.

        Args:
            state (int): The current state of the environment.

        Returns:
            int: The selected action.
        """
        if np.random.random() < self.epsilon:  # Checks for random exploration
            action = np.random.randint(0, self.environment.num_actions)  # Randomly selects an action
        else:
            action = np.argmax(self.Q_table[state, :])  # Selects the action with the highest Q-value

        return action

    def learn(self, state, action, reward, next_state):
        """
        Updates the Q-table based on the experience tuple (state, action, reward, next_state).

        Args:
            state (int): The current state.
            action (int): The action taken in the current state.
            reward (float): The reward received for taking the action.
            next_state (int): The next state after taking the action.
        """
        Q_value = self.Q_table[state, action]  # Retrieves the current Q-value for the state-action pair
        max_Q_value = np.max(self.Q_table[next_state, :])  # Calculates the maximum Q-value for the next state
        new_Q_value = reward + self.gamma * max_Q_value  # Calculates the updated Q-value

        self.Q_table[state, action] = new_Q_value  # Updates the Q-table with the new Q-value

    def run(self, num_episodes):
        """
        Trains the agent for a specified number of episodes.

        Args:
            num_episodes (int): The number of episodes to train for.
        """
        for episode in range(num_episodes):
            state = self.environment.reset()  # Resets the environment
            done = False  # Initializes the done flag

            while not done:
                action = self.act(state)  # Selects an action based on the current state
                next_state, reward, done = self.environment.step(action)  # Takes the action and observes the outcome

                self.learn(state, action, reward, next_state)  # Updates the Q-table based on the experience tuple
                state = next_state  # Updates the current state

    def process_input(self, user_input):
        """
        Processes user input and generates a response.

        Args:
            user_input (str): The input from the user.

        Returns:
            str: The agent's response.
        """
        # Simple keyword matching for responses
        if "hello" in user_input.lower():
            return "Hello! How can I assist you today?"
        elif "how are you" in user_input.lower():
            return "I'm just a virtual assistant, but I'm here to help you!"
        elif "bye" in user_input.lower():
            return "Goodbye! Have a great day!"
        else:
            return "I'm sorry, I don't understand that. Can you please rephrase?"


