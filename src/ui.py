class UserInterface:
    def __init__(self, agent):
        self.agent = agent

    def start(self):
        print("Mr. Data is now active. Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Shutting down Mr. Data. Goodbye!")
                break
            response = self.agent.process_input(user_input)
            print(f"Mr. Data: {response}")
