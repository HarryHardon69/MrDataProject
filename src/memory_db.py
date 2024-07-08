# src/memory_db.py
import os

def list_files(directory):
    return os.listdir(directory)

def read_file(filepath):
    with open(filepath, 'r') as file:
        return file.read()

def write_file(filepath, content):
    with open(filepath, 'w') as file:
        file.write(content)

def delete_file(filepath):
    os.remove(filepath)
