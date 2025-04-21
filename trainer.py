import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset, random_split

from model_utils import TinyWakeWordModel


class MemoryEfficientDataset(Dataset):
    """Memory-efficient dataset that processes data in chunks."""

    def __init__(self, X, y):
        """
        Initialize the dataset with features and labels.

        Args:
            X: Training features (can be a list or array)
            y: Training labels (can be a list or array)
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert to torch tensors here, not earlier to save memory
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0)
        return features, label


def train_model_efficient(X_train, y_train, device, epochs=15, batch_size=64,
                          val_split=0.2, patience=5, checkpoint_dir="model_checkpoints"):
    """
    Train the wake word detection model with improved memory efficiency and checkpointing.

    Args:
        X_train: Training features
        y_train: Training labels
        device: Torch device
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        val_split: Validation split ratio (0-1)
        patience: Early stopping patience
        checkpoint_dir: Directory to save model checkpoints

    Returns:
        Trained model
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create dataset with memory-efficient loading
    full_dataset = MemoryEfficientDataset(X_train, y_train)

    # Get size of each split
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    # Create training and validation splits
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    print(f"Training set: {len(train_dataset)} examples")
    print(f"Validation set: {len(val_dataset)} examples")

    # Create data loaders with pinned memory for faster GPU transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),  # Only use pinned memory if GPU is available
        num_workers=0  # Adjust based on your system, 0 for no multiprocessing
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0
    )

    # Calculate class weights for imbalanced data
    positive_samples = sum(y_train)
    negative_samples = len(y_train) - positive_samples
    pos_weight = negative_samples / max(1, positive_samples)  # Avoid division by zero

    print(f"Class weight for positive samples: {pos_weight:.2f}")

    # Create model
    model = TinyWakeWordModel().to(device)

    # Use binary cross entropy loss with class weights
    pos_weight_tensor = torch.tensor([pos_weight]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Use AdamW optimizer with weight decay to prevent overfitting
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.002,
        weight_decay=0.01,
        amsgrad=True  # More stable optimization
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # Early stopping parameters
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")

    # Training loop
    print("Training wake word model...")

    # Track metrics for each epoch
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    # Ensure gradients are zeroed to start
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Use tqdm for progress bar
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
            # Move tensors to the right device
            inputs, labels = inputs.to(device), labels.to(device)

            # Clear accumulated gradients
            optimizer.zero_grad(set_to_none=True)  # set_to_none is more memory efficient

            # Forward pass
            logits = model(inputs)
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimize
            optimizer.step()

            # Update statistics
            train_loss += loss.item() * inputs.size(0)  # Scale by batch size
            predicted = (torch.sigmoid(logits) > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Explicitly free memory
            del inputs, labels, logits, loss, predicted
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # No need to track gradients
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                logits = model(inputs)
                loss = criterion(logits, labels)

                # Update statistics
                val_loss += loss.item() * inputs.size(0)
                predicted = (torch.sigmoid(logits) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Explicitly free memory
                del inputs, labels, logits, loss, predicted
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Calculate average metrics
        avg_train_loss = train_loss / train_total
        train_accuracy = 100 * train_correct / train_total
        avg_val_loss = val_loss / val_total
        val_accuracy = 100 * val_correct / val_total

        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_accuracy)
        history["lr"].append(current_lr)

        # Print epoch statistics
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0

            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_acc': val_accuracy,
                'history': history
            }, best_model_path)

            print(f"  New best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            early_stop_counter += 1
            print(f"  Early stopping counter: {early_stop_counter}/{patience}")

            # Also save