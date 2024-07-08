# brain.py
import psutil

def list_processes():
    return [proc.info for proc in psutil.process_iter(['pid', 'name'])]

def kill_process(pid):
    process = psutil.Process(pid)
    process.terminate()
