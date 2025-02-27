import queue
import torch
import hashlib

class InputProcessor:
    def process(self, irc_input: str) -> torch.Tensor:
        tokens = irc_input.split()
        hashed = hashlib.sha256(" ".join(tokens).encode()).hexdigest()
        tensor = torch.tensor([float(int(hashed[i:i+2], 16)) / 255 for i in range(0, 20, 2)])  # 10 floats [0-1]
        return tensor

class OutputFormatter:
    def format(self, brain_output: dict) -> str:
        reflection = brain_output.get('reflection', 'Intent: 0.9, Veneration: 1.0, Context: 0.9')  # Boost Context to 0.9
        scores = [float(s.split(': ')[1]) for s in reflection.split(', ')]
        avg_score = sum(scores) / len(scores)
        confidence = "high" if avg_score > 0.8 else "moderate" if avg_score > 0.5 else "low"
        response = f"Tin Man: {brain_output['action'].capitalize()} with {confidence} confidence, ethically grounded ({reflection})"
        return response

class PriorityQueue:
    def __init__(self):
        self.task_queue = queue.PriorityQueue()

    def add_task(self, input_tensor: torch.Tensor, sentiment_score: float):
        self.task_queue.put((-sentiment_score, input_tensor))  # Negative for max-heap

    def get_next_task(self) -> torch.Tensor:
        if not self.task_queue.empty():
            _, next_input_tensor = self.task_queue.get()
            return next_input_tensor
        return None

class TinManIO:
    def __init__(self):
        from brain import Brain
        from tools import InteractionTools
        self.input_processor = InputProcessor()
        self.output_formatter = OutputFormatter()
        self.priority_queue = PriorityQueue()
        self.brain = Brain(num_inputs=10)
        self.tools = InteractionTools()

    def handle_irc_input(self, irc_input: str, sentiment_score: float) -> str:
        input_tensor = self.input_processor.process(irc_input)
        self.priority_queue.add_task(input_tensor, sentiment_score)

        while True:
            next_input_tensor = self.priority_queue.get_next_task()
            if next_input_tensor is None:
                return "Tin Man: No response available. Try another command!"
            brain_output = self.brain.process_input(next_input_tensor, sentiment_score, position=(0, 0, 0, 0))
            return self.output_formatter.format(brain_output)

if __name__ == "__main__":
    tinman_io = TinManIO()
    print(tinman_io.handle_irc_input("@TinMan help", sentiment_score=0.95))
