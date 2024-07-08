# src/ui.py
import tkinter as tk

def create_ui():
    root = tk.Tk()
    root.title("Mr. Data Interface")

    label = tk.Label(root, text="Welcome to Mr. Data Interface")
    label.pack()

    root.mainloop()
