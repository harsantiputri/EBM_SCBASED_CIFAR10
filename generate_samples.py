# generate_samples.py

import torch

from models.ebm_cnn import EBM_CNN
from training.trainer import EBM_Trainer
from data.cifar10_dataset import CIFAR10Dataset
from config import DEVICE, MODEL_DIR, SAMPLING_STEPS, SAMPLING_STEP_SIZE

def main():
    print(f"Using device: {DEVICE}")

    # 1. Setup Dataset
    print("Setting up CIFAR-10 dataset for configuration...")
    dataset_handler = CIFAR10Dataset()
    _, _ = dataset_handler.get_dataloaders()
    print("Dataset configuration loaded.")

    # 2. Initialize Model
    print("Initializing EBM_CNN model...")
    model = EBM_CNN(num_channels=3, image_size=32)
    print("Model initialized.")

    # 3. Initialize Trainer (to use its loading and sampling methods)
    print("Initializing Trainer...")
    # We pass dummy loaders as they aren't used for sample generation directly in this script
    trainer = EBM_Trainer(model, None, None)

    # 4. Load the trained model
    model_path = os.path.join(MODEL_DIR, "ebm_cifar10.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first using 'train.py'.")
        return

    print(f"Loading trained model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully.")

    # 5. Generate Samples
    print("Generating samples from the trained model...")
    # You can adjust num_samples, step_size, and num_steps here
    num_samples_to_generate = 64 # Generate 64 samples
    trainer.generate_samples(
        num_samples=num_samples_to_generate,
        step_size=SAMPLING_STEP_SIZE, # Use the configured step size
        num_steps=SAMPLING_STEPS      # Use the configured number of steps
    )
    print("Sample generation complete.")

if __name__ == "__main__":
    import os # Need to import os here as well for path joining
    main()