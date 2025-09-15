# training/trainer.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import os
import torchvision.utils as vutils

from models.ebm_cnn import EBM_CNN
from config import (
    DEVICE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,
    TRAIN_SIGMA, SAVE_INTERVAL, PRINT_INTERVAL, SAMPLES_DIR, MODEL_DIR, LOG_DIR,
    SAMPLING_INTERVAL, SAMPLING_STEPS, SAMPLING_STEP_SIZE, IMAGE_SIZE, NUM_CHANNELS
)

class EBM_Trainer:
    def __init__(self, model, train_loader, test_loader=None):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.criterion = nn.MSELoss() # Mean Squared Error loss for predicting noise

        # Setup logging and saving paths
        self.model_save_path = os.path.join(MODEL_DIR, f"ebm_cifar10.pth")
        self.sample_save_dir = SAMPLES_DIR

        self.current_epoch = 0

    def calculate_loss(self, images, sigma):
        # Denoising Score Matching (DSM) objective:
        # We train the model to predict the noise that was added to the data.
        # Add noise to the data
        noise = torch.randn_like(images) * sigma
        noisy_images = images + noise

        # Predict the noise using the model
        # In a more general setting, the model might take sigma as input too.
        predicted_noise = self.model(noisy_images, sigma)

        # Calculate the loss (MSE between actual noise and predicted noise)
        loss = self.criterion(predicted_noise, noise)
        return loss

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        # Use tqdm for a progress bar
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{NUM_EPOCHS}", leave=False)

        for i, (images, _) in enumerate(progress_bar):
            images = images.to(DEVICE)

            # Calculate loss for the current sigma
            # In a multi-sigma setting, we'd sample sigmas here.
            sigma = torch.tensor(TRAIN_SIGMA).to(DEVICE)
            loss = self.calculate_loss(images, sigma)

            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Print interval
            if (i + 1) % PRINT_INTERVAL == 0:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / (i + 1):.4f}'
                })

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_model(self):
        print(f"Saving model to {self.model_save_path}")
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.model_save_path)

    def load_model(self):
        if os.path.exists(self.model_save_path):
            print(f"Loading model from {self.model_save_path}")
            checkpoint = torch.load(self.model_save_path, map_location=DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint['epoch']
            print(f"Model loaded. Resuming from epoch {self.current_epoch+1}")
        else:
            print("No existing model found. Starting training from scratch.")

    def generate_samples(self, num_samples=16, step_size=SAMPLING_STEP_SIZE, num_steps=SAMPLING_STEPS):
        """Generates samples using Langevin Dynamics."""
        self.model.eval() # Ensure model is in evaluation mode

        # Start with random noise
        samples = torch.randn(num_samples, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

        # Precompute sigma for Langevin dynamics.
        # In a more sophisticated approach, this sigma would be part of the training/sampling strategy.
        # For simplicity here, we'll use a fixed sigma for the entire sampling process.
        # We use a fixed sigma for simplicity.
        sigma_score = torch.tensor(TRAIN_SIGMA).to(DEVICE) # Use the training sigma for score scaling

        print(f"Generating {num_samples} samples using Langevin Dynamics...")
        print(f"Sampling parameters: step_size={step_size}, num_steps={num_steps}, sigma_score={sigma_score.item():.4f}")

        for step in tqdm(range(num_steps), desc="Langevin Sampling", leave=False):
            samples.requires_grad_(True)
            with torch.no_grad(): # During inference, we don't need gradients for the model itself
                 predicted_noise = self.model(samples) # Predict noise

            # Calculate the score from the predicted noise
            score = predicted_noise / sigma_score # Assuming noise ~ N(0, sigma^2 I)      

            # To avoid issues with potential division by zero or very small sigmas,
            # and to keep it simpler, let's assume the model directly outputs something proportional to the score.
            # A common approach in simpler score-based models is to directly use the model output as a gradient step.

            # Update rule: x_{t+1} = x_t + step_size * score(x_t) + sqrt(2*step_size) * z
            # where score(x_t) = -predicted_noise / sigma_score^2
            # score_term = - (step_size * predicted_noise) / (sigma_score ** 2)
            # noise_term = torch.randn_like(samples) * torch.sqrt(torch.tensor(2 * step_size))
            # simplified "denoising" update. We use the predicted noise directly as the gradient.
            denoising_step_size = 1e-4 # this is new parameter reduced to 1e-4 from 0.1
            update_direction = predicted_noise
            samples = samples - denoising_step_size * update_direction
            #samples in non-denoising context
            #samples = samples + score_term + noise_term

            # Clip samples to stay within the normalized range [-1, 1]
            samples = torch.clamp(samples, -1.0, 1.0)

        # De-normalize and save samples
        samples = (samples + 1) / 2 # Scale back to [0, 1]
        samples = torch.clamp(samples, 0.0, 1.0)
        save_path = os.path.join(self.sample_save_dir, f"epoch_{self.current_epoch+1}_samples.png")
        vutils.save_image(samples, save_path, nrow=int(num_samples**0.5), normalize=False)
        print(f"Saved samples to {save_path}")


    def train(self):
        self.load_model() # Try to load a pre-trained model

        for epoch in range(self.current_epoch, NUM_EPOCHS):
            self.current_epoch = epoch
            avg_loss = self.train_epoch()
            print(f"Epoch {self.current_epoch+1}/{NUM_EPOCHS} | Average Loss: {avg_loss:.4f}")

            # Save model periodically
            if (epoch + 1) % SAVE_INTERVAL == 0:
                self.save_model()

            # Generate and save samples periodically
            if (epoch + 1) % SAMPLING_INTERVAL == 0:
                self.generate_samples()

        # Save the final model
        self.save_model()
        print("Training finished.")