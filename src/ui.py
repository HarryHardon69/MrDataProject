# src/ui.py
import tkinter as tk
from brain import Brain

class DataUI:
    def __init__(self):
        self.brain = Brain()

    def create_ui(self):
        root = tk.Tk()
        root.title("Mr. Data Interface")

        label = tk.Label(root, text="Welcome to Mr. Data Interface")
        label.pack()

        start_button = tk.Button(root, text="Start", command=self.start_data)
        start_button.pack()

        root.mainloop()

    def start_data(self):
        self.brain.initialize()
        self.brain.run()

if __name__ == "__main__":
    ui = DataUI()
    ui.create_ui()
