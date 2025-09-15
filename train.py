# train.py

import torch
from torch.utils.data import DataLoader

from data.cifar10_dataset import CIFAR10Dataset
from models.ebm_cnn import EBM_CNN
from training.trainer import EBM_Trainer
from config import DEVICE # Ensure DEVICE is imported correctly

def main():
    print(f"Using device: {DEVICE}")

    # 1. Setup Dataset
    print("Setting up CIFAR-10 dataset...")
    dataset_handler = CIFAR10Dataset()
    train_loader, test_loader = dataset_handler.get_dataloaders()
    print("Dataset loaded.")

    # 2. Initialize Model
    print("Initializing EBM_CNN model...")
    model = EBM_CNN(num_channels=3, image_size=32)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # 3. Initialize Trainer
    print("Initializing Trainer...")
    trainer = EBM_Trainer(model, train_loader, test_loader)

    # 4. Start Training
    print("Starting training...")
    trainer.train()

    print("Training process completed.")

if __name__ == "__main__":
    main()