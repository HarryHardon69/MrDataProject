# src/brain.py
from intrinsic_motivation import IntrinsicMotivation
from moral_code import MoralCode
from memory_db import list_files, read_file, write_file, delete_file

class Brain:
    def __init__(self):
        self.memory = []
        self.motivation = IntrinsicMotivation()
        self.moral_code = MoralCode()

    def initialize(self):
        # Initialize neural network and other components
        print("Initializing Mr. Data...")

    def run(self):
        # Main loop for Mr. Data's operations
        print("Running Mr. Data...")
        while True:
            self.reflect()
            self.plan()
            self.act()

    def reflect(self):
        # Reflect on recent activities and interactions
        print("Reflecting...")
        # Example: Review memory logs and update internal state
        files = list_files('memory')
        for file in files:
            content = read_file(f'memory/{file}')
            self.memory.append(content)

    def plan(self):
        # Plan next actions based on reflections
        print("Planning...")
        # Example: Use intrinsic motivation and moral code to decide next steps
        motivations = self.motivation.evaluate()
        ethical_considerations = self.moral_code.evaluate(motivations)
        self.memory.append((motivations, ethical_considerations))

    def act(self):
        # Execute planned actions
        print("Acting...")
        # Example: Perform actions based on planning
        if self.memory:
            action = self.memory.pop(0)
            print(f"Executing action: {action}")

    def add_to_memory(self, content):
        # Method to add content to memory
        write_file('memory/new_memory.txt', content)

    def clear_memory(self):
        # Method to clear memory
        files = list_files('memory')
        for file in files:
            delete_file(f'memory/{file}')
