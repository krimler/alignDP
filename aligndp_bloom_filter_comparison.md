# The Privacy-Utility Trade-Off: A Parametric Study of ALIGNDP

## Project Overview

This repository explores the performance of the ALIGNDP (ALignment-driven INtelligent Differential Privacy) algorithm, a selective differential privacy (DP) mechanism for feedback learning systems. ALIGNDP's key innovation is its ability to apply DP only to rare, privacy-sensitive data, thus optimizing the classic privacy-utility trade-off.

## Core Hypothesis

The hypothesis for this study is that ALIGNDP's selective approach allows for a superior privacy-utility trade-off curve compared to uniform DP methods. We can achieve a higher level of data utility for a given level of privacy, or maintain a consistent level of utility with a stronger privacy guarantee. This is achieved by spending the privacy budget (`epsilon`) only where it is truly needed.

## Experiment 2: Fine-Tuning the Privacy Parameter ($\epsilon$) with Bloom Filter

### Objective

The objective of this experiment is to demonstrate the impact of the privacy budget ($\epsilon$) on model learning efficiency and data utility. We will compare the performance of ALIGNDP against a uniform LDP method (RAPPOR) across a range of `epsilon` values. This experiment uses a more realistic implementation of ALIGNDP's storage mechanism by including a Bloom Filter for regular feedback.

### Experiment Design

The script runs a simulation of LLM feedback collection for `N` users. We use three different `epsilon` values for the experiment: a strong privacy guarantee ($\epsilon=0.1$), a moderate one ($\epsilon=1.0$), and a weaker one ($\epsilon=5.0$).

For each `epsilon`, we will:
1.  Run the **RAPPOR_Simulator** using `epsilon` as the uniform privacy budget.
2.  Run the **ALIGNDP_Algorithm** using `epsilon` *only for the rare feedback*. Regular feedback is logged to a Bloom Filter without privacy distortion.

### Key Performance Metrics

* **Relative Error:** The difference between the original, true counts and the reported noisy counts for each feedback category.
* **Total Relative Error:** The aggregate error across all feedback.
* **Privacy Budget ($\epsilon$):** The value used to control the amount of noise. A smaller $\epsilon$ means stronger privacy and more noise.

### How to Run

1.  **Dependencies:** This experiment requires the `bitarray` library to run the Bloom Filter implementation.
    ```bash
    pip install numpy bitarray
    ```
2.  **Execution:** Run the script named `aligndp_bloom_filter_comparison.py` from your terminal.
    ```bash
    python aligndp_bloom_filter_comparison.py
    ```

### Expected Results

The output will clearly show the following trends:

* **For Regular Feedback (`dislike`, `override`):** ALIGNDP will consistently produce a relative error of near zero, regardless of the `epsilon` value. The uniform LDP method will show an error that is inversely proportional to `epsilon`.
* **For Rare Feedback (`like`):** Both methods will show an error that is inversely proportional to `epsilon`. However, due to its focused budget, ALIGNDP will often yield a better signal (lower error) for a given `epsilon` because it is not "wasting" the budget on other events.
* **Overall Utility:** The `Total Relative Error` for ALIGNDP will be significantly lower across all `epsilon` values, empirically proving that it offers a superior privacy-utility trade-off.
