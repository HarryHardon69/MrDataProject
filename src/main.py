from brain import MrDataAgent
from ui import UserInterface
import time

def main():
    # Initialize the environment and agent
    environment = None  # Placeholder: Define or import your environment
    agent = MrDataAgent(environment)
    ui = UserInterface()

    # Main loop
    try:
        while True:
            user_input = ui.get_user_input()
            agent_response = agent.process_input(user_input)
            ui.display_response(agent_response)
            
            # Sleep for a short duration to simulate real-time processing
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Shutting down Mr. Data Agent.")

if __name__ == "__main__":
    main()
