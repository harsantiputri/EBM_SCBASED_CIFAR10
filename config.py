import os
import torch

# Working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# --- Dataset Configuration ---
DATASET_NAME = "CIFAR10"
BATCH_SIZE = 128
NUM_WORKERS = 4
IMAGE_SIZE = 32
NUM_CHANNELS = 3

# --- Model Configuration ---
# Using a simple CNN architecture.
# The output of the network is expected to be the "score" (gradient of log-density)
# or the predicted noise.
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100 # Reduced for faster demonstration

# --- Training Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_INTERVAL = 10
SAVE_INTERVAL = 10
SAMPLING_INTERVAL = 20

# --- Score Matching Specific Configuration ---
# We'll use Denoising Score Matching (DSM)
# This requires adding noise to the data.
# Different noise levels (sigmas) are often used.
# For simplicity, we'll start with a fixed sigma and potentially a schedule.
SIGMA_MIN = 0.01
SIGMA_MAX = 1.0
NUM_NOISE_LEVELS = 10 # Number of noise levels to consider for training
# A common practice is to use a geometric progression for sigmas.
# SIGMAS = torch.logspace(torch.log(torch.tensor(SIGMA_MIN)), torch.log(torch.tensor(SIGMA_MAX)), NUM_NOISE_LEVELS)
# For this example, we'll start with a single noise level for simplicity and then
# can extend to multiple levels.
TRAIN_SIGMA = 0.5

# --- Langevin Dynamics Configuration (for sampling) ---
SAMPLING_STEPS = 5000 # upped to 5000 from 100, and 500 consecutively
SAMPLING_STEP_SIZE = 1e-5 # decrease drastically from 0.05