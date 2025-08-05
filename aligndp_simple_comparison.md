# ALIGNDP: A Selective Differential Privacy Algorithm for Feedback Learning

## Project Overview

This repository contains the code for the ALIGNDP (ALignment-driven INtelligent Differential Privacy) algorithm. ALIGNDP is a novel server-side differential privacy (DP) mechanism designed to improve the privacy-utility trade-off in systems that learn from user feedback.

Unlike existing Local Differential Privacy (LDP) methods like RAPPOR, which apply uniform noise to all data, ALIGNDP applies DP selectively. It identifies and privatizes only "rare" or anomalous feedback events, while utilizing regular feedback directly without distortion. This approach aims to provide a higher-quality signal for model training while consuming less privacy budget.

## Core Hypothesis

The central hypothesis of this project is that by selectively applying DP, ALIGNDP can achieve superior learning efficiency and privacy efficiency compared to uniform LDP methods. We demonstrate this by simulating a real-world scenario of collecting user feedback for a Large Language Model (LLM).

## Experiment 1: ALIGNDP vs. Uniform LDP (Core Concept)

### Objective

The goal of this experiment is to directly compare the data utility and privacy budget consumption of the ALIGNDP algorithm against a simplified uniform LDP method (similar to RAPPOR). The experiment uses synthetic LLM feedback data, where "like" events are designated as rare and "dislike" and "override" events are considered regular.

This version of the experiment focuses on the fundamental difference: ALIGNDP provides a perfect signal for regular events while RAPPOR adds noise to all events.

### Algorithm Implementations

* **RAPPOR_Simulator:** A simplified uniform LDP mechanism. It applies a randomized response to every feedback event, regardless of its type, introducing noise across the entire dataset.
* **ALIGNDP_Algorithm:** Our proposed method. It uses an anomaly detector to identify rare events (`"like"`) and applies Laplace noise only to these events. Regular feedback (`"dislike"`, `"override"`) is processed without noise to preserve the full signal. For simplicity in this experiment, a Python `Counter` is used to represent the storage of regular feedback, demonstrating that the full signal is retained.

### How to Run

1.  **Dependencies:** Ensure you have the required Python libraries installed:
    ```bash
    pip install numpy
    ```
    (Note: The `bitarray` library is not required for this experiment.)
2.  **Execution:** Run the script named `aligndp_simple_comparison.py` from your terminal.
    ```bash
    python aligndp_simple_comparison.py
    ```

### Expected Results

The script will run a series of comparisons across different `epsilon` values.

* **RAPPOR_Simulator:** The results will show noise and distortion across all feedback counts. The relative error for all categories, including the frequent ones, will be non-zero.
* **ALIGNDP_Algorithm:** The results will show near-perfect counts for the regular feedback (`"dislike"`, `"override"`) with a relative error close to zero. The rare feedback (`"like"`) will have some controlled noise, but because the privacy budget is focused, the signal-to-noise ratio is better than in the RAPPOR simulation.

This experiment provides empirical evidence that ALIGNDP effectively separates privacy-sensitive data from non-sensitive data, leading to a significant improvement in overall data utility for model training.
