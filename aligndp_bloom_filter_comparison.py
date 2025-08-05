import numpy as np
import collections
import random
import hashlib
from bitarray import bitarray
from typing import List, Dict, Any

# --- Helper Classes and Functions ---

class UserFeedback:
    """Represents a single user feedback event."""
    def __init__(self, prompt: str, response: str, feedback: str):
        self.prompt = prompt
        self.response = response
        self.feedback = feedback

class BloomFilter:
    """A simple implementation of a Bloom filter."""
    def __init__(self, size: int, hash_count: int):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(self.size)
        self.bit_array.setall(0)

    def _hashes(self, item: str):
        """Generates multiple hash values for an item."""
        hashes = []
        for i in range(self.hash_count):
            hash_result = int(hashlib.md5(f"{item}{i}".encode()).hexdigest(), 16) % self.size
            hashes.append(hash_result)
        return hashes

    def add(self, item: str):
        """Adds an item to the filter."""
        for hash_value in self._hashes(item):
            self.bit_array[hash_value] = 1

    def check(self, item: str) -> bool:
        """Checks if an item is possibly in the filter."""
        return all(self.bit_array[hash_value] == 1 for hash_value in self._hashes(item))

class AnomalyDetector:
    """A simple anomaly detector for rare feedback."""
    def __init__(self, rare_events: set):
        self.rare_events = rare_events

    def is_rare(self, feedback: str) -> bool:
        return feedback in self.rare_events

def apply_laplace_noise(value: int, epsilon: float) -> int:
    """Applies Laplace noise for differential privacy."""
    sensitivity = 1
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0, scale=scale)
    return int(value + noise)

def calculate_utility(original_counts: collections.Counter, noisy_data: List[str]) -> Dict[str, Any]:
    """Calculates the utility (accuracy) of the privatized data."""
    noisy_counts = collections.Counter(noisy_data)
    
    accuracy = {}
    for key, count in original_counts.items():
        noisy_count = noisy_counts.get(key, 0)
        if count > 0:
            error = abs(count - noisy_count) / count
        else:
            error = "N/A"
        accuracy[key] = {"original": count, "noisy": noisy_count, "relative_error": error}
    
    total_original = sum(original_counts.values())
    total_noisy = len(noisy_data)
    
    if total_original > 0:
        total_error = abs(total_original - total_noisy) / total_original
    else:
        total_error = "N/A"
    
    return {"per_category_accuracy": accuracy, "total_relative_error": total_error}

# --- Privacy Algorithm Implementations ---

class RAPPOR_Simulator:
    """
    A simplified RAPPOR-like simulator.
    Applies uniform randomized response to all events.
    """
    def __init__(self, epsilon: float, vocab: list):
        self.epsilon = epsilon
        self.vocab = vocab
        self.privatized_data = []

    def privatize(self, feedback: str):
        p = np.exp(self.epsilon) / (np.exp(self.epsilon) + len(self.vocab) - 1)
        
        if random.random() < p:
            self.privatized_data.append(feedback)
        else:
            other_values = [v for v in self.vocab if v != feedback]
            self.privatized_data.append(random.choice(other_values))

class ALIGNDP_Algorithm:
    """
    Your ALIGNDP algorithm, applying DP only to rare events.
    """
    def __init__(self, rare_events: set, epsilon_rare: float, regular_feedback_bloom_size: int, bloom_hash_count: int):
        self.rare_events = rare_events
        self.epsilon_rare = epsilon_rare
        self.detector = AnomalyDetector(rare_events)
        self.privatized_data = []
        self.regular_feedback_bloom = BloomFilter(regular_feedback_bloom_size, bloom_hash_count)
    
    def process_feedback(self, feedback: str):
        if self.detector.is_rare(feedback):
            noisy_count = apply_laplace_noise(1, self.epsilon_rare)
            for _ in range(max(0, noisy_count)):
                self.privatized_data.append(feedback)
        else:
            self.regular_feedback_bloom.add(feedback)
            self.privatized_data.append(feedback)

# --- Simulation and Comparison ---

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

def run_comparison(num_users: int, epsilon_to_test: float, rare_events: set):
    """
    Runs a comparison of the algorithms for a given epsilon.
    For ALIGNDP, this epsilon is used for the rare events.
    """
    FEEDBACK_VOCAB = ["like", "dislike", "override"]

    print(f"--- Simulating LLM Feedback Data for {num_users} users ---")
    data = simulate_llm_interaction(num_users)
    original_feedback_counts = collections.Counter(f.feedback for f in data)
    print(f"Original Feedback Counts: {original_feedback_counts}")
    print(f"Epsilon for test: {epsilon_to_test}")
    print("-" * 40)
    
    # 1. RAPPOR-like comparison
    rappor = RAPPOR_Simulator(epsilon=epsilon_to_test, vocab=FEEDBACK_VOCAB)
    for f in data:
        rappor.privatize(f.feedback)
    rappor_utility = calculate_utility(original_feedback_counts, rappor.privatized_data)
    print("RAPPOR-like Results (Uniform LDP):")
    for key, val in rappor_utility['per_category_accuracy'].items():
        print(f"  - {key}: Original={val['original']}, Noisy={val['noisy']}, Error={val['relative_error']:.2f}")
    print(f"  Total Relative Error: {rappor_utility['total_relative_error']:.2f}")
    print("-" * 40)
    
    # 2. ALIGNDP algorithm
    aligndp = ALIGNDP_Algorithm(
        rare_events=rare_events,
        epsilon_rare=epsilon_to_test,
        regular_feedback_bloom_size=1000,
        bloom_hash_count=5
    )
    for f in data:
        aligndp.process_feedback(f.feedback)
    aligndp_utility = calculate_utility(original_feedback_counts, aligndp.privatized_data)
    print("ALIGNDP Results (Selective DP):")
    for key, val in aligndp_utility['per_category_accuracy'].items():
        print(f"  - {key}: Original={val['original']}, Noisy={val['noisy']}, Error={val['relative_error']:.2f}")
    print(f"  Total Relative Error: {aligndp_utility['total_relative_error']:.2f}")
    print("-" * 40)

def main():
    """Main function to run the complete comparison."""
    NUM_USERS = 10000
    RARE_EVENT_TYPES = {"like"}
    
    print("--- ALIGNDP Algorithm vs. Uniform LDP Comparison ---")
    
    epsilon_values_to_test = [0.1, 1.0, 5.0]
    
    for epsilon in epsilon_values_to_test:
        run_comparison(num_users=NUM_USERS, epsilon_to_test=epsilon, rare_events=RARE_EVENT_TYPES)

if __name__ == "__main__":
    main()
