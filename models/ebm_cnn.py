# models/ebm_cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class EBM_CNN(nn.Module):
    def __init__(self, num_channels=3, image_size=32):
        super(EBM_CNN, self).__init__()
        self.num_channels = num_channels
        self.image_size = image_size

        # --- Simple CNN Architecture ---
        # Input: (batch_size, num_channels, image_size, image_size)

        # Layer 1
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Layer 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # Downsample
        self.bn2 = nn.BatchNorm2d(128)

        # Layer 3
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Layer 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # Downsample
        self.bn4 = nn.BatchNorm2d(256)

        # Layer 5
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        # Layer 6 - Upsampling to predict noise of the same shape as input
        # Image size after two stride=2 operations: image_size / 4
        # We need to upsample back to image_size.
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # Upsample
        self.bndeconv1 = nn.BatchNorm2d(128)

        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1) # Upsample
        self.bndeconv2 = nn.BatchNorm2d(128)

        self.deconv3 = nn.Conv2d(128, num_channels, kernel_size=3, stride=1, padding=1) # Final prediction

    def forward(self, x, sigma=None):
        # For this simple CNN, we'll predict noise regardless of sigma value,
        # assuming the training data is generated with a specific sigma.

        # Layer 1
        x = F.relu(self.bn1(self.conv1(x)))
        # Layer 2
        x = F.relu(self.bn2(self.conv2(x)))
        # Layer 3
        x = F.relu(self.bn3(self.conv3(x)))
        # Layer 4
        x = F.relu(self.bn4(self.conv4(x)))
        # Layer 5
        x = F.relu(self.bn5(self.conv5(x)))

        # Upsampling to match input dimensions
        # Layer 6 (Upsample)
        x = F.relu(self.bndeconv1(self.deconv1(x)))
        # Layer 7 (Upsample)
        x = F.relu(self.bndeconv2(self.deconv2(x)))
        # Final layer: Predict noise/score
        # The output should have the same shape as the input
        noise_pred = self.deconv3(x) # Output is the predicted noise

        return noise_pred

    def predict_score(self, x, sigma):
        # In Denoising Score Matching, the model predicts the noise.
        # The score is approximately -noise / sigma^2.
        # We'll directly output the noise prediction for the loss calculation.
        # In a more sophisticated setup, sigma would be an input to the network.
        return self.forward(x)


if __name__ == '__main__':
    # Example usage
    print("Testing EBM_CNN model...")
    model = EBM_CNN(num_channels=3, image_size=32)
    model.eval() # Set model to evaluation mode

    # Create a dummy input tensor
    # (batch_size, num_channels, height, width)
    dummy_input = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        predicted_noise = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output (predicted noise) shape: {predicted_noise.shape}")
    assert predicted_noise.shape == dummy_input.shape, "Output shape does not match input shape!"

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    print("EBM_CNN model test complete.")