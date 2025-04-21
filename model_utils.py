"""
Model utilities for wake word detection.
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
import config


class TinyWakeWordModel(nn.Module):
    """
    Ultra-lightweight wake word detection model designed for ESP32-S3.
    Uses minimal parameters and quantization-friendly operations.
    """

    def __init__(self, input_size=config.INPUT_FEATURES, hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS):
        super(TinyWakeWordModel, self).__init__()

        # 1. Feature extraction with small-footprint 1D convolutions
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # 2. Temporal processing with GRU (lighter than LSTM)
        self.lstm = nn.GRU(
            input_size=16,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # Unidirectional for efficiency
            dropout=0.2 if num_layers > 1 else 0
        )

        # 3. Simplified attention mechanism
        self.attention = nn.Linear(hidden_size, 1)

        # 4. Final classification layers with bottleneck
        self.fc1 = nn.Linear(hidden_size, 8)
        self.fc2 = nn.Linear(8, 1)

        # Using efficient activations
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input shape: batch_size x time_steps x features

        # Transpose for 1D convolution (batch, channels, length)
        x = x.transpose(1, 2)  # Shape becomes: batch_size x features x time_steps

        # Apply convolution and pooling
        x = self.relu(self.conv1(x))  # Shape: batch_size x 16 x time_steps
        x = self.pool(x)  # Shape: batch_size x 16 x (time_steps/2)

        # Transpose back for LSTM (batch, time_steps, channels)
        x = x.transpose(1, 2)  # Shape: batch_size x (time_steps/2) x 16

        # Apply LSTM/GRU - returns all hidden states and final hidden state
        lstm_out, _ = self.lstm(x)  # lstm_out shape: batch_size x (time_steps/2) x hidden_size

        # Apply simplified attention
        # Calculate attention weights
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)

        # Create context vector as weighted sum of hidden states
        context = torch.sum(attention_weights * lstm_out, dim=1)  # Shape: batch_size x hidden_size

        # Final classification
        x = self.relu(self.fc1(context))  # Shape: batch_size x 8
        x = self.fc2(x)  # Shape: batch_size x 1 (logits, not probabilities)

        return x


def load_model(model_path=config.MODEL_PATH):
    """
    Load the wake word detection model.

    Args:
        model_path: Path to the model file

    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        # Check if it's a TorchScript model or regular PyTorch model
        try:
            # Try loading as TorchScript first
            model = torch.jit.load(model_path, map_location='cpu')
            print("Loaded TorchScript model")
        except Exception as e:
            print(f"Not a TorchScript model, loading as regular PyTorch model: {e}")
            # Fall back to loading as regular model
            model = TinyWakeWordModel()
            model.load_state_dict(torch.load(model_path, map_location='cpu'))

        model.eval()  # Set model to evaluation mode
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        raise


class WakeWordDetector:
    """
    Wake word detector that processes audio and runs inference with the model.
    """

    def __init__(self, model_path=config.MODEL_PATH):
        """
        Initialize wake word detector.

        Args:
            model_path: Path to the model file
        """
        self.model = load_model(model_path)
        self.threshold = config.DETECTION_THRESHOLD
        self.positive_count = 0
        self.required_positives = config.ACTIVATION_FRAMES
        self.last_detection_time = 0
        self.cooldown_period = 2.0  # seconds

    def process_audio(self, audio):
        """
        Process audio and detect wake word.

        Args:
            audio: Audio signal array

        Returns:
            True if wake word detected, False otherwise
        """
        # Check if we're in cooldown period
        current_time = time.time()
        if current_time - self.last_detection_time < self.cooldown_period:
            return False

        try:
            # Extract features
            features = extract_mfcc_features(audio)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

            # Run inference
            with torch.no_grad():
                output = self.model(features_tensor)
                prediction = torch.sigmoid(output).item()

            # Update detection state
            if prediction > self.threshold:
                self.positive_count += 1
                print(
                    f"Potential wake word detected! Confidence: {prediction:.4f}, Count: {self.positive_count}/{self.required_positives}")
            else:
                self.positive_count = max(0, self.positive_count - 1)  # Decay count if not detected

            # Check if we have enough consecutive detections
            if self.positive_count >= self.required_positives:
                self.last_detection_time = current_time
                self.positive_count = 0
                return True

            return False

        except Exception as e:
            print(f"Error processing audio: {e}")
            return False


# Import this here to avoid circular imports
from audio_utils import extract_mfcc_features