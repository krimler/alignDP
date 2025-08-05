import numpy as np
import collections
import random
from typing import List, Dict, Any

# --- Step 1: Data Model ---
class UserFeedback:
    """Represents a single user feedback event."""
    def __init__(self, prompt: str, response: str, feedback: str):
        self.prompt = prompt
        self.response = response
        self.feedback = feedback

# --- Step 2: Anomaly Detection and Differential Privacy Helpers ---
class AnomalyDetector:
    """A simple anomaly detector for rare feedback."""
    def __init__(self, rare_events: set):
        self.rare_events = rare_events

    def is_rare(self, feedback: str) -> bool:
        return feedback in self.rare_events

def apply_laplace_noise(value: int, epsilon: float) -> int:
    """Applies Laplace noise for differential privacy."""
    sensitivity = 1  # For a simple counting query
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0, scale=scale)
    return int(value + noise)

# --- Step 3: Implement Privacy Algorithms ---

class RAPPOR_Simulator:
    """
    A simplified RAPPOR-like simulator.
    Applies uniform randomized response to all events.
    """
    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        self.privatized_data = []

    def privatize(self, feedback: str):
        # A simple randomized response:
        # with probability 1/(e^epsilon + 1) we flip the bit
        # This is a very simplified version for demonstration
        if random.random() < 1 / (np.exp(self.epsilon) + 1):
            # "Noisy" flip - simulate an inaccurate report
            flipped_feedback = "dislike" if feedback == "like" else "like"
            self.privatized_data.append(flipped_feedback)
        else:
            # Report honestly
            self.privatized_data.append(feedback)

class Apple_Privacy_Simulator:
    """
    A simplified Apple-like LDP simulator.
    Uses randomized response but for a limited vocabulary.
    """
    def __init__(self, epsilon: float, vocab: list):
        self.epsilon = epsilon
        self.vocab = vocab
        self.privatized_data = []

    def privatize(self, feedback: str):
        # A simplified randomized response mechanism
        p = np.exp(self.epsilon) / (np.exp(self.epsilon) + len(self.vocab) - 1)
        q = 1 / (np.exp(self.epsilon) + len(self.vocab) - 1)

        if random.random() < p:
            # Report truthfully
            self.privatized_data.append(feedback)
        else:
            # Report a random value from the vocab (excluding the true value)
            other_values = [v for v in self.vocab if v != feedback]
            self.privatized_data.append(random.choice(other_values))

class ALIGNDP_Algorithm:
    """
    Your ALIGNDP algorithm, applying DP only to rare events.
    """
    def __init__(self, rare_events: set, epsilon_rare: float, epsilon_regular: float = np.inf):
        self.rare_events = rare_events
        self.epsilon_rare = epsilon_rare
        self.epsilon_regular = epsilon_regular
        self.detector = AnomalyDetector(rare_events)
        self.privatized_data = []
        self.regular_feedback = collections.Counter()
    
    def process_feedback(self, feedback: str):
        if self.detector.is_rare(feedback):
            # Rare event: apply DP.
            # Here, we're simply adding noise to a count of the event.
            # In a real implementation, this would be more complex.
            noisy_count = apply_laplace_noise(1, self.epsilon_rare)
            if noisy_count > 0:
                self.privatized_data.append(feedback)
            else:
                # If noise makes it negative, we can ignore the event
                pass
        else:
            # Regular event: no DP, just log.
            self.regular_feedback[feedback] += 1
            self.privatized_data.append(feedback)

def calculate_utility(original_counts: collections.Counter, noisy_data: List[str]) -> Dict[str, Any]:
    """Calculates the utility (accuracy) of the privatized data."""
    noisy_counts = collections.Counter(noisy_data)
    
    accuracy = {}
    for key, count in original_counts.items():
        noisy_count = noisy_counts.get(key, 0)
        # We can measure accuracy as the relative error
        if count > 0:
            error = abs(count - noisy_count) / count
        else:
            error = "N/A"
        accuracy[key] = {"original": count, "noisy": noisy_count, "relative_error": error}
    
    # Calculate overall error
    total_original = sum(original_counts.values())
    total_noisy = len(noisy_data)
    
    total_error = abs(total_original - total_noisy) / total_original
    
    return {"per_category_accuracy": accuracy, "total_relative_error": total_error}

# --- Step 4: Simulation and Comparison ---

def simulate_llm_interaction(num_users: int) -> List[UserFeedback]:
    """Simulates LLM chat interactions."""
    feedback_types = ["dislike", "dislike", "dislike", "override", "override", "like"]
    prompts = ["Tell me about machine learning.", "Write a poem about the sea."]
    responses = ["Machine learning is a field of AI...", "The ocean breathes in tides of blue..."]
    
    feedback_data = []
    for _ in range(num_users):
        feedback = random.choice(feedback_types)
        prompt = random.choice(prompts)
        response = random.choice(responses)
        feedback_data.append(UserFeedback(prompt, response, feedback))
        
    return feedback_data

def run_comparison():
    # Simulation parameters
    NUM_USERS = 1000
    RARE_EVENT_TYPES = {"like"}
    FEEDBACK_VOCAB = ["like", "dislike", "override"]
    EPSILON_RAPPOR = 2.0  # Common epsilon value for LDP
    EPSILON_APPLE = 2.0
    EPSILON_ALIGNDP_RARE = 0.5  # A tighter budget for a rare event
    
    # Generate synthetic data
    print("--- Simulating LLM Feedback Data ---")
    data = simulate_llm_interaction(NUM_USERS)
    original_feedback_counts = collections.Counter(f.feedback for f in data)
    print(f"Original Feedback Counts: {original_feedback_counts}")
    print("-" * 40)
    
    # 1. RAPPOR-like comparison
    print("--- Running RAPPOR-like Simulator ---")
    rappor = RAPPOR_Simulator(epsilon=EPSILON_RAPPOR)
    for f in data:
        rappor.privatize(f.feedback)
    rappor_utility = calculate_utility(original_feedback_counts, rappor.privatized_data)
    print("RAPPOR Results:")
    print(rappor_utility)
    print("-" * 40)
    
    # 2. Apple-like LDP comparison
    print("--- Running Apple-like Simulator ---")
    apple_priv = Apple_Privacy_Simulator(epsilon=EPSILON_APPLE, vocab=FEEDBACK_VOCAB)
    for f in data:
        apple_priv.privatize(f.feedback)
    apple_utility = calculate_utility(original_feedback_counts, apple_priv.privatized_data)
    print("Apple-like LDP Results:")
    print(apple_utility)
    print("-" * 40)

    # 3. Your ALIGNDP algorithm
    print("--- Running ALIGNDP Algorithm ---")
    aligndp = ALIGNDP_Algorithm(rare_events=RARE_EVENT_TYPES, epsilon_rare=EPSILON_ALIGNDP_RARE)
    for f in data:
        aligndp.process_feedback(f.feedback)
    aligndp_utility = calculate_utility(original_feedback_counts, aligndp.privatized_data)
    print("ALIGNDP Results:")
    print(aligndp_utility)
    print("-" * 40)
    
if __name__ == "__main__":
    run_comparison()
