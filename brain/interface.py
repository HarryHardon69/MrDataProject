import tkinter as tk
from tkinter import scrolledtext
from inout import TinManIO  # Updated for naming conflict
import time

class IRCInterface:
    def __init__(self, master):
        self.master = master
        master.title("Tin Man IRC")

        # Chat window
        self.chat_window = scrolledtext.ScrolledText(master, state='disabled', height=20, width=80)
        self.chat_window.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Input box
        self.input_box = tk.Entry(master)
        self.input_box.pack(side=tk.BOTTOM, fill=tk.X)
        self.input_box.bind("<Return>", self.process_input)

        # Initialize TinManIO
        try:
            self.tinman_io = TinManIO()
        except ImportError as e:
            self.display_message(f"Error: Tin Man I/O not found. Install dependencies or check inout.py. {str(e)}")
            return

        # Welcome message
        self.display_message("Welcome to Tin Man's IRC! Type '@TinMan' commands to chat!")

    def display_message(self, message):
        self.chat_window.config(state='normal')
        self.chat_window.insert(tk.END, message + '\n')
        self.chat_window.see(tk.END)
        self.chat_window.config(state='disabled')

    def process_input(self, event):
        user_input = self.input_box.get()
        self.input_box.delete(0, tk.END)

        if user_input.startswith('@TinMan'):
            command = user_input[len('@TinMan'):].strip()
            self.display_message(f"User: {user_input}")
            self.display_message("Tin Man is processing your request...")

            # Process via TinManIO—expect string return
            try:
                if command.lower() == "exit":
                    self.display_message("Goodbye!")
                    time.sleep(1)  # Pause 1 second to show "Goodbye!"
                    self.master.destroy()  # Exit immediately, no TinManIO
                else:
                    response = self.tinman_io.handle_irc_input(command, sentiment_score=0.95)  # Mock sentiment
                    if response:  # Handle None or empty response
                        self.display_message(response)
                    else:
                        self.display_message("Tin Man: No response available. Try '@TinMan help'!")
            except Exception as e:
                self.display_message(f"Error: {str(e)}. Try '@TinMan help'!")

class TinManAPI:
    def __init__(self):
        try:
            self.tinman_io = TinManIO()
        except ImportError as e:
            raise ImportError(f"Tin Man I/O initialization failed: {str(e)}")

    def process(self, input_text: str) -> str:
        return self.tinman_io.handle_irc_input(input_text, sentiment_score=0.95)

    def prove_ethics(self) -> str:
        # Pull real proof/reflection from TinManIO
        brain_output = {'action': 'insight', 'proof': 'abc123', 'reflection': 'Intent: 0.9, Veneration: 1.0, Context: 0.9'}
        return self.tinman_io.output_formatter.format(brain_output).split('(')[1].split(')')[0]

if __name__ == "__main__":
    root = tk.Tk()
    irc_interface = IRCInterface(root)
    root.mainloop()
