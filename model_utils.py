"""
Model utilities for wake word detection.
"""

import os
import torch
import torch.nn as nn
import config

from config import DETECTION_THRESHOLD


class TinyWakeWordModel(nn.Module):
    """
    Enhanced lightweight wake word detection model for ESP32-S3.
    Slightly larger capacity while maintaining efficiency.
    """

    def __init__(self, input_size=config.INPUT_FEATURES, hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS):
        super(TinyWakeWordModel, self).__init__()

        # 1. Enhanced feature extraction with dual convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=24,  # Increased from 16
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=24,
            out_channels=32,  # Added second conv layer
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.batch_norm1 = nn.BatchNorm1d(24)  # Added batch normalization
        self.batch_norm2 = nn.BatchNorm1d(32)

        # 2. Improved temporal processing
        self.gru = nn.GRU(
            input_size=32,  # Increased to match conv output
            hidden_size=hidden_size * 2,  # Doubled hidden size
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # Still unidirectional for efficiency
            dropout=0.3 if num_layers > 1 else 0  # Slightly increased dropout
        )

        # 3. Multi-head attention mechanism (simplified version)
        self.attention_heads = 2
        self.attention = nn.Linear(hidden_size * 2, self.attention_heads)

        # 4. Enhanced classification with wider layers
        self.fc1 = nn.Linear(hidden_size * 2, 32)  # Widened bottleneck
        self.fc2 = nn.Linear(32, 16)  # Added intermediate layer
        self.fc3 = nn.Linear(16, 1)  # Output layer

        # Using different activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.2)  # Added dropout between FC layers

    def forward(self, x):
        # Input shape: batch_size x time_steps x features

        # Transpose for 1D convolution (batch, channels, length)
        x = x.transpose(1, 2)  # Shape: batch_size x features x time_steps

        # Apply enhanced convolution blocks with batch norm
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.pool(x)  # Pooling after both convolutions

        # Transpose back for GRU (batch, time_steps, channels)
        x = x.transpose(1, 2)  # Shape: batch_size x (time_steps/2) x 32

        # Apply GRU
        gru_out, _ = self.gru(x)  # gru_out shape: batch_size x (time_steps/2) x (hidden_size*2)

        # Apply multi-head attention
        attention_weights = torch.softmax(self.attention(gru_out), dim=1)  # Shape: batch x time x heads

        # Calculate context vectors (one per head)
        contexts = []
        for h in range(self.attention_heads):
            head_weights = attention_weights[:, :, h].unsqueeze(2)  # batch x time x 1
            context = torch.sum(head_weights * gru_out, dim=1)  # batch x hidden
            contexts.append(context)

        # Concatenate context vectors
        context = torch.cat(contexts, dim=1) if self.attention_heads > 1 else contexts[0]

        # If we have multiple heads, average them instead of concatenating to maintain dimension
        if self.attention_heads > 1:
            context = torch.mean(torch.stack(contexts), dim=0)

        # Enhanced classification path
        x = self.leaky_relu(self.fc1(context))  # Shape: batch_size x 32
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))  # Shape: batch_size x 16
        x = self.fc3(x)  # Shape: batch_size x 1

        return x


def load_model(model_path=config.MODEL_PATH):
    """
    Load the wake word detection model for continued training.

    Args:
        model_path: Path to the model file

    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        # Always create a fresh instance of the model architecture
        fresh_model = TinyWakeWordModel()

        try:
            # If it's a TorchScript model
            loaded_model = torch.jit.load(model_path, map_location='cpu')
            print("Loaded TorchScript model")

            # Copy parameters from TorchScript model to the fresh model
            with torch.no_grad():
                for name, param in fresh_model.named_parameters():
                    # Find corresponding parameter in the TorchScript model
                    # The parameter access can differ between regular and TorchScript models
                    try:
                        # Try the direct access first
                        src_param = getattr(loaded_model, name)
                        param.copy_(src_param)
                    except (AttributeError, RuntimeError):
                        # If that fails, we need to extract the parameter differently
                        print(f"Could not directly copy parameter {name}, using alternative method")
                        # This part may need adjustment based on how your TorchScript model is structured
                        for src_name, src_param in loaded_model.named_parameters():
                            if src_name.endswith(name.split('.')[-1]):
                                if param.shape == src_param.shape:
                                    param.copy_(src_param)
                                    break

            return fresh_model

        except Exception as e:
            print(f"Not a TorchScript model, loading as regular PyTorch model: {e}")
            # For regular PyTorch model, just load the state dict
            fresh_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            return fresh_model

    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def load_model_eval(model_path=config.MODEL_PATH):
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
        self.model = load_model_eval(model_path)
        self.threshold = config.DETECTION_THRESHOLD
        self.required_positives = config.ACTIVATION_FRAMES

    def process_audio(self, audio):
        """
        Process audio and detect wake word.

        Args:
            audio: Audio signal array

        Returns:
            True if wake word detected, False otherwise
        """

        try:
            # Extract features
            features = extract_mfcc_features(audio)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

            # Run inference
            with torch.no_grad():
                output = self.model(features_tensor)
                prediction = torch.sigmoid(output).item()

            level = int(prediction * 100)
            # Update detection state
            if prediction > self.threshold:
                print("X"*level)
                return True
            else:
                achieved = "." * level
                missing = "." * (int(DETECTION_THRESHOLD * 100 )- level)
                maxi = "-" * (100 - int(DETECTION_THRESHOLD * 100))
                bar =  f"{achieved}X{missing}|{maxi}"
                print(bar)

            return False

        except Exception as e:
            print(f"Error processing audio: {e}")
            return False


# Import this here to avoid circular imports
from audio_utils import extract_mfcc_features