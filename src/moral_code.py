# src/moral_code.py
class MoralCode:
    def __init__(self):
        self.values = {
            "honesty": 1.0,
            "empathy": 0.8,
            "justice": 0.9
        }

    def evaluate_action(self, action):
        # Evaluate action based on moral values
        score = 0
        for value in self.values:
            score += self.values[value] * action.get(value, 0)
        return score

    def decide(self, possible_actions):
        # Decide best action based on moral evaluation
        best_action = None
        best_score = -float('inf')
        for action in possible_actions:
            score = self.evaluate_action(action)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action
