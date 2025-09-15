# EBM with Score Matching on CIFAR-10: An Experiment Journey

This repository documents an experiment in building a simple Energy-Based Model (EBM) using a score-based method (Denoising Score Matching) on the CIFAR-10 dataset.

The primary goal was not to achieve state-of-the-art results, but to understand the fundamentals and debug the common failure modes of this class of generative models.

### Project Structure

```
EBM_001_SCBASED_CIFAR10/
├── data/
├── models/
│   └── ebm_cnn.py
├── training/
│   └── trainer.py
├── utils/
│   └── utils.py
├── experiment_outputs/
│   ├── epoch_60_sampler_fail_noisy.png
│   ├── denoising_test_fail_black.png
│   └── denoising_test_fail_stable_noise.png
├── config.py
├── train.py
├── generate_samples.py
├── .gitignore
├── README.md
└── requirements.txt
```

### Setup and Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/YourUsername/YourRepoName.git
    cd YourRepoName
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  To train the model:
    ```bash
    python train.py
    ```

---

### The Experiment Journey: From Training to Failure Analysis

The model was trained successfully, and the loss function showed a steady decrease, indicating that the network was learning its objective.

#### Initial Sampling Attempt (Epoch 60)

After 60 epochs of training, the first attempt at generating samples using Langevin Dynamics was made.

**Result:** The samples were indistinguishable from random noise.
![Initial Noisy Samples](experiment_outputs/epoch_60_sampler_fail_noisy.png)

**Diagnosis:** The hyperparameters for the Langevin sampler were unstable. The random "noise term" was several orders of magnitude stronger than the model's "score term" (guidance), causing the sampler to perform a random walk.

#### Diagnostic Test 1: Gradient Descent (Denoising)

To isolate the model's guidance, the random noise term in the sampler was removed, effectively turning it into a gradient descent process on the energy function.

**Result:** The samples immediately collapsed to a solid black color.
![Black Samples](experiment_outputs/denoising_test_fail_black.png)

**Diagnosis:** This proved the model's guidance was having a powerful, systematic effect. However, the step size was catastrophically large, causing the pixel values to "explode" and be clamped to -1 (which corresponds to black).

#### Diagnostic Test 2: Stable Gradient Descent

The step size for the gradient descent test was drastically reduced to prevent the explosion.

**Result:** The samples remained as random noise, showing no change from their initial state.
![Stable Noisy Samples](experiment_outputs/denoising_test_fail_stable_noise.png)

**Diagnosis:** This was the key insight. When the sampler is stable, the model's guidance is too weak to have any effect. The model learned to minimize its loss but failed to learn a useful representation of the data's true gradient field.

### Conclusion and Next Steps

The experiments successfully demonstrated that the model failed to learn a useful generative function due to two primary reasons:

1.  **Single Noise Level:** The model was only trained at `sigma=0.5` and was therefore unable to provide useful guidance at other noise levels encountered during the sampling process.
2.  **Limited Model Capacity:** The simple CNN architecture was likely not powerful enough to capture the complex, multi-scale distribution of the CIFAR-10 dataset.

The clear path forward is to evolve this project into a proper **Denoising Diffusion Model** by:
1.  Implementing training across a continuous schedule of noise levels.
2.  Upgrading the model architecture to a time-aware **U-Net**.
