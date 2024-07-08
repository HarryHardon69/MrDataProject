from src.brain import MrDataAgent
from src.environment import Environment
from src.ui import UserInterface

def main():
    # Initialize environment and agent
    env = Environment()
    agent = MrDataAgent(env)
    
    # Initialize UI
    ui = UserInterface(agent)
    
    # Start interaction loop
    ui.start()

if __name__ == "__main__":
    main()
