import numpy as np
import collections
import random
import csv
from typing import List, Dict, Any

# --- Helper Classes and Functions ---

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
    # Use all keys from both original and noisy counts to cover all cases
    all_keys = set(original_counts.keys()) | set(noisy_counts.keys())
    
    for key in sorted(all_keys):
        original_count = original_counts.get(key, 0)
        noisy_count = noisy_counts.get(key, 0)
        
        if original_count > 0:
            error = abs(original_count - noisy_count) / original_count
        else:
            error = "N/A" # Handle cases where a category has a count of zero
        
        accuracy[key] = {"original": original_count, "noisy": noisy_count, "relative_error": error}
    
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
        # A simpler randomized response for a small vocabulary
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
    def __init__(self, rare_events: set, epsilon_rare: float):
        self.rare_events = rare_events
        self.epsilon_rare = epsilon_rare
        self.detector = AnomalyDetector(rare_events)
        self.privatized_data = []
    
    def process_feedback(self, feedback: str):
        if self.detector.is_rare(feedback):
            noisy_count = apply_laplace_noise(1, self.epsilon_rare)
            # Add the rare event with noise to the list
            for _ in range(max(0, noisy_count)):
                self.privatized_data.append(feedback)
        else:
            # Regular event: no DP.
            self.privatized_data.append(feedback)

# --- Data Loading from CSV ---

def load_feedback_data_from_csv(filename: str) -> List[str]:
    """
    Loads feedback data from a CSV file using the built-in csv module.
    Assumes a column named 'Feedback' exists.
    """
    feedback_list = []
    try:
        with open(filename, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                feedback_list.append(row['Feedback'])
        print(f"Successfully loaded {len(feedback_list)} entries from '{filename}'.")
        return feedback_list
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return []
    except KeyError:
        print(f"Error: The 'Feedback' column was not found in '{filename}'.")
        return []

# --- Experiment Runner ---

def run_csv_comparison(filename: str, epsilon_to_test: float, rare_events: set):
    """
    Orchestrates the full comparison using data from a local CSV file.
    """
    print("--- ALIGNDP Algorithm vs. Uniform LDP Comparison on Local Data ---")
    
    feedback_data = load_feedback_data_from_csv(filename=filename)
    
    if not feedback_data:
        return
        
    # The vocabulary must be a list of all possible feedback types in your data.
    # The code now uses the actual categories from your output.
    FEEDBACK_VOCAB = list(set(feedback_data))
    original_feedback_counts = collections.Counter(feedback_data)
    
    print(f"\nOriginal Feedback Counts: {original_feedback_counts}")
    print(f"Running comparison with epsilon = {epsilon_to_test}")
    print("-" * 40)
    
    # 1. RAPPOR-like comparison
    rappor = RAPPOR_Simulator(epsilon=epsilon_to_test, vocab=FEEDBACK_VOCAB)
    for f in feedback_data:
        rappor.privatize(f)
    rappor_utility = calculate_utility(original_feedback_counts, rappor.privatized_data)
    print("RAPPOR-like Results (Uniform LDP):")
    for key, val in rappor_utility['per_category_accuracy'].items():
        print(f"  - {key}: Original={val['original']}, Noisy={val['noisy']}, Error={val['relative_error']:.2f}")
    print(f"  Total Relative Error: {rappor_utility['total_relative_error']:.2f}")
    print("-" * 40)
    
    # 2. ALIGNDP algorithm
    aligndp = ALIGNDP_Algorithm(rare_events=rare_events, epsilon_rare=epsilon_to_test)
    for f in feedback_data:
        aligndp.process_feedback(f)
    aligndp_utility = calculate_utility(original_feedback_counts, aligndp.privatized_data)
    print("ALIGNDP Results (Selective DP):")
    for key, val in aligndp_utility['per_category_accuracy'].items():
        print(f"  - {key}: Original={val['original']}, Noisy={val['noisy']}, Error={val['relative_error']:.2f}")
    print(f"  Total Relative Error: {aligndp_utility['total_relative_error']:.2f}")
    print("-" * 40)

'''
if __name__ == "__main__":
    FILENAME = "align_dp_feedback_data.csv"
    EPSILON_TEST = 1.0
    # Update this set to define which feedback types are considered "rare"
    RARE_EVENT_TYPES = {'override'}
    
    run_csv_comparison(filename=FILENAME, epsilon_to_test=EPSILON_TEST, rare_events=RARE_EVENT_TYPES)
'''
def run_multiple_simulations(num_runs: int, filename: str, epsilon: float, rare_events: set):
    rappor_total_errors = collections.defaultdict(list)
    aligndp_total_errors = collections.defaultdict(list)

    print(f"\n--- Running {num_runs} simulations to get average results ---")
    
    for i in range(num_runs):
        feedback_data = load_feedback_data_from_csv(filename=filename)
        
        if not feedback_data:
            print("Failed to load data. Aborting simulations.")
            return

        FEEDBACK_VOCAB = list(set(feedback_data))
        original_counts = collections.Counter(feedback_data)

        # RAPPOR Simulation
        rappor = RAPPOR_Simulator(epsilon=epsilon, vocab=FEEDBACK_VOCAB)
        for f in feedback_data:
            rappor.privatize(f)
        rappor_utility = calculate_utility(original_counts, rappor.privatized_data)
        for key, val in rappor_utility['per_category_accuracy'].items():
            rappor_total_errors[key].append(val['relative_error'])
        rappor_total_errors['total'].append(rappor_utility['total_relative_error'])

        # ALIGNDP Simulation
        aligndp = ALIGNDP_Algorithm(rare_events=rare_events, epsilon_rare=epsilon)
        for f in feedback_data:
            aligndp.process_feedback(f)
        aligndp_utility = calculate_utility(original_counts, aligndp.privatized_data)
        for key, val in aligndp_utility['per_category_accuracy'].items():
            aligndp_total_errors[key].append(val['relative_error'])
        aligndp_total_errors['total'].append(aligndp_utility['total_relative_error'])

    print("\n--- Average Results over {} runs ---".format(num_runs))
    
    print("RAPPOR-like (Uniform LDP) Average Errors:")
    for key, errors in rappor_total_errors.items():
        if key != 'total':
            print(f"  - {key}: {np.mean([e for e in errors if e != 'N/A']):.4f}")
    print(f"  Total Average Error: {np.mean([e for e in rappor_total_errors['total'] if e != 'N/A']):.4f}")
    
    print("\nALIGNDP (Selective DP) Average Errors:")
    for key, errors in aligndp_total_errors.items():
        if key != 'total':
            print(f"  - {key}: {np.mean([e for e in errors if e != 'N/A']):.4f}")
    print(f"  Total Average Error: {np.mean([e for e in aligndp_total_errors['total'] if e != 'N/A']):.4f}")

# Update the main block to call the new function
if __name__ == "__main__":
    FILENAME = "align_dp_feedback_data.csv"
    EPSILON_TEST = 1.0
    RARE_EVENT_TYPES = {'override'}
    NUM_SIMULATIONS = 100
    
    run_multiple_simulations(NUM_SIMULATIONS, FILENAME, EPSILON_TEST, RARE_EVENT_TYPES)

