import os
import json
import time
import torch
from tqdm import tqdm
import numpy as np


from model_utils import TinyWakeWordModel, load_model
from teacher import check_device, load_model_and_processor
from datapreprocessor import WakewordProcessor, create_training_data
from config import INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, WINDOW_SIZE_MS, HOP_LENGTH_MS, SAMPLE_RATE, WAKEWORD, \
    SAMPLES, MODEL_ID, MLS_PATH, LOCAL_RECORDINGS, NO_CACHE, MODEL_PATH


def optimize_memory_settings():
    """
    Optimize memory settings for large dataset processing.

    Sets appropriate PyTorch memory settings and configures process limits
    to avoid memory issues with large datasets.
    """
    # Set PyTorch memory allocation strategy
    if torch.cuda.is_available():
        # Configure PyTorch to release memory when possible
        torch.cuda.empty_cache()

        # Try to enable memory caching to reuse allocations
        try:
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use at most 90% of GPU memory
            print("GPU memory allocation limited to 90%")
        except:
            print("Could not set GPU memory fraction, using default")

        # Print available GPU memory
        try:
            free_memory, total_memory = torch.cuda.mem_get_info()
            print(f"GPU memory: {free_memory / 1024 ** 3:.2f}GB free of {total_memory / 1024 ** 3:.2f}GB total")
        except:
            print("Could not query GPU memory info")

    # Configure Python garbage collection for better memory management
    import gc
    gc.set_threshold(100, 5, 5)  # More aggressive garbage collection

    # Optimize numpy memory usage
    import numpy as np
    np.set_printoptions(precision=4, suppress=True, threshold=100)

    # System-level optimizations if possible
    try:
        import resource
        # Increase soft limit to the hard limit
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            resource.setrlimit(resource.RLIMIT_AS, (hard, hard))
            print(f"Memory limit increased from {soft / 1024 ** 3:.1f}GB to {hard / 1024 ** 3:.1f}GB")
    except:
        print("Could not adjust system resource limits")


def train_model(X_train, y_train, device, epochs=30, batch_size=64, continue_training=False):
    """
    Train the wake word detection model with validation split and early stopping.

    Args:
        X_train: Training features
        y_train: Training labels
        device: Torch device
        epochs: Maximum number of training epochs
        batch_size: Batch size for training

    Returns:
        Trained model
    """
    # Split data into training and validation sets (80/20 split)
    split = int(0.8 * len(X_train))
    indices = np.random.permutation(len(X_train))

    train_indices = indices[:split]
    val_indices = indices[split:]

    X_train_split = X_train[train_indices]
    y_train_split = y_train[train_indices]
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]

    print(f"Training set: {len(X_train_split)} examples")
    print(f"Validation set: {len(X_val)} examples")

    # Create datasets and dataloaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train_split, dtype=torch.float32),
        torch.tensor(y_train_split, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # Create model
    lr = 0.01
    if continue_training:
        model = load_model()
        model.to(device)
        model.train()
        lr = 0.005
        print("Loaded model")
    else:
        model = TinyWakeWordModel().to(device)

    # Use binary cross entropy loss with class weights to handle imbalanced data
    pos_weight = torch.tensor([3.0]).to(device)  # Adjust based on class distribution
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Use AdamW optimizer with learning rate schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=6, verbose=True
    )

    # Early stopping parameters
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 10
    best_model_state = None

    # Training loop
    print("Training wake word model...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(inputs)
            outputs = torch.sigmoid(logits)
            loss = criterion(logits, labels)

            # Backward pass and optimize
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Statistics
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                logits = model(inputs)
                outputs = torch.sigmoid(logits)
                loss = criterion(logits, labels)

                # Statistics
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate average metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        # Print epoch statistics
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            # Save best model state
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  New best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            early_stop_counter += 1
            print(f"  Early stopping counter: {early_stop_counter}/{early_stop_patience}")

            if early_stop_counter >= early_stop_patience:
                print("Early stopping triggered!")
                break

    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)
        print(f"Loaded best model with validation loss: {best_val_loss:.4f}")

    # Calculate final performance metrics
    model.eval()
    tp, fp, tn, fn = 0, 0, 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = torch.sigmoid(model(inputs))
            predicted = (outputs > 0.5).float()

            tp += ((predicted == 1) & (labels == 1)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            tn += ((predicted == 0) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\nFinal Model Performance:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return model


def export_for_esp32(model, output_dir="esp32_model"):
    """
    Export model for ESP32-S3 deployment with comprehensive optimization.

    Args:
        model: Trained PyTorch model
        output_dir: Output directory for the exported model
    """

    os.makedirs(output_dir, exist_ok=True)

    # Set model to evaluation mode
    model.eval()

    print("Optimizing and exporting model for ESP32-S3...")

    # Move model to CPU before quantization (to avoid CUDA quantization issues)
    cpu_model = model.cpu()

    # 1. Save the non-quantized model first (as backup)
    original_model_path = os.path.join(output_dir, "wakeword_model_original.pt")
    torch.save(cpu_model.state_dict(), original_model_path)
    print(f"Original model saved to {original_model_path}")

    try:
        # 2. Quantize the model to 8-bit for significant size reduction
        #    Quantization must be done on CPU model
        quantized_model = torch.quantization.quantize_dynamic(
            cpu_model,
            {torch.nn.Linear, torch.nn.Conv1d, torch.nn.GRU},
            dtype=torch.qint8
        )

        # 3. Create example input for tracing (representative of runtime input)
        example_input = torch.randn(1, 50, INPUT_FEATURES)  # Batch size 1, 50 time steps, 13 features

        # 4. Convert to TorchScript for deployment
        try:
            # Try script mode first (handles dynamic control flow better)
            script_model = torch.jit.script(quantized_model)
            print("Successfully created script model")
        except Exception as e:
            print(f"Script mode failed: {e}")
            print("Falling back to trace mode")
            # Fall back to tracing if scripting fails
            script_model = torch.jit.trace(quantized_model, example_input)

        # 5. Optimize the TorchScript model for inference
        optimized_model = torch.jit.optimize_for_inference(script_model)

        # 6. Save the model

        optimized_model.save(MODEL_PATH)
        print(f"Quantized model exported to {MODEL_PATH}")
        model_size = os.path.getsize(MODEL_PATH) / 1024  # KB

    except Exception as e:
        print(f"Quantization failed: {e}")
        print("Saving non-quantized model instead")
        # If quantization fails, export the regular model
        torch.save(cpu_model.state_dict(), MODEL_PATH)
        print(f"Non-quantized model exported to {MODEL_PATH}")
        model_size = os.path.getsize(MODEL_PATH) / 1024  # KB

    # 7. Export model architecture and parameters as C header for direct embedding
    model_header_path = os.path.join(output_dir, "wakeword_model.h")

    # Generate C code (simplified example)
    with open(model_header_path, 'w') as f:
        f.write("// Auto-generated wake word model parameters\n")
        f.write("#ifndef WAKEWORD_MODEL_H\n")
        f.write("#define WAKEWORD_MODEL_H\n\n")

        # Add model configuration
        f.write(f"#define WAKEWORD_INPUT_FEATURES {INPUT_FEATURES}\n")
        f.write(f"#define WAKEWORD_HIDDEN_SIZE {HIDDEN_SIZE}\n")
        f.write(f"#define WAKEWORD_NUM_LAYERS {NUM_LAYERS}\n")
        f.write(f"#define WAKEWORD_WINDOW_SIZE_MS {WINDOW_SIZE_MS}\n")
        f.write(f"#define WAKEWORD_HOP_LENGTH_MS {HOP_LENGTH_MS}\n")
        f.write(f"#define WAKEWORD_SAMPLE_RATE {SAMPLE_RATE}\n\n")

        # Add feature extraction parameters
        f.write(f"#define WAKEWORD_FFT_SIZE {512}\n")
        f.write(f"#define WAKEWORD_MEL_BANDS {40}\n\n")

        # Add detection threshold
        f.write(f"#define WAKEWORD_DETECTION_THRESHOLD 0.7f\n\n")

        f.write("#endif // WAKEWORD_MODEL_H\n")

    print(f"Model header file exported to {model_header_path}")

    # 8. Save model metadata in JSON format for tooling
    metadata = {
        "input_features": INPUT_FEATURES,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "window_size_ms": WINDOW_SIZE_MS,
        "hop_length_ms": HOP_LENGTH_MS,
        "sample_rate": SAMPLE_RATE,
        "fft_size": 512,
        "mel_bands": 40,
        "model_size_bytes": os.path.getsize(MODEL_PATH),
        "detection_threshold": 0.7
    }

    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Model metadata exported to {metadata_path}")

    # 9. Print model size information
    print(f"Model size: {model_size:.2f} KB")

    return MODEL_PATH


def load_or_create_training_data(cache_dir, device, dataset_path = MLS_PATH):
    """
    Loads training data from cache if available, or creates and caches it if not.

    Args:
        cache_dir (str): Directory for cached files
        device (torch.device): Device to use for processing

    Returns:
        tuple: (X_train, y_train, dataset_info)
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Generate a cache key based on relevant parameters
    cache_key = f"{SAMPLES}_{dataset_path.replace('/', '-')}"
    training_data_cache_path = os.path.join(cache_dir, f"training_data_{cache_key}.npz")

    # Try to load training data from cache
    if os.path.exists(training_data_cache_path) and not NO_CACHE:
        print(f"Loading training data from cache: {training_data_cache_path}")
        try:
            data = np.load(training_data_cache_path, allow_pickle=True)
            X_train = data['X_train']
            y_train = data['y_train']
            dataset_info = data['dataset_info'].item() if 'dataset_info' in data else {}
            print(f"Training data loaded from cache ({len(X_train)} examples)")
            return X_train, y_train, dataset_info
        except Exception as e:
            print(f"Error loading cache: {e}. Will regenerate training data.")

    print(f"Cache miss, creating training data: {training_data_cache_path}")
    # If we get here, we need to generate the data
    teacher_processor, teacher_model = load_model_and_processor(device)

    # Initialize the wakeword processor
    processor = WakewordProcessor(dataset_path, teacher_model, teacher_processor, device)

    # Process the dataset
    print(f"Processing dataset with wakeword: '{WAKEWORD}'")
    start_time = time.time()
    dataset, wakeword_locations = processor.process_dataset(
        include_negatives=True
    )
    process_time = time.time() - start_time
    print(f"Dataset processing completed in {process_time:.2f} seconds")

    # Create training data
    print("Creating training data from processed dataset...")
    X_train, y_train = create_training_data(dataset, wakeword_locations)

    # Create dataset info for summary
    dataset_info = {
        "total_files_processed": len(dataset),
        "files_with_wakeword": sum(1 for locs in wakeword_locations.values() if locs),
        "process_time_seconds": process_time
    }

    # Save training data to cache
    print(f"Saving training data to cache: {training_data_cache_path}")
    np.savez_compressed(
        training_data_cache_path,
        X_train=X_train,
        y_train=y_train,
        dataset_info=dataset_info
    )

    return X_train, y_train, dataset_info


def main():
    """Main function with memory-efficient wakeword model training."""
    # Configuration
    CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

    DATASOURCE = LOCAL_RECORDINGS
    # DATASOURCE = MLS_PATH

    # Check device
    device = check_device()

    # Load or create training data with caching
    X_train, y_train, dataset_info = load_or_create_training_data(
        cache_dir=CACHE_DIR,
        device=device,
        dataset_path=DATASOURCE
    )

    # Check if we have enough data for training
    if len(X_train) < 100:
        print(f"Not enough training data. Only found {len(X_train)} examples.")
        if sum(y_train == 1.0) < 10:
            print(f"Too few positive examples. Only found {sum(y_train == 1.0)}.")
            print("Consider using a different wakeword or adding more data.")
            return

    # Train model
    print("Training wakeword detection model...")
    model = train_model(X_train, y_train, device, continue_training=True)

    # Export model for ESP32-S3
    print("Exporting model for ESP32-S3...")
    model_path = export_for_esp32(model)

    print(f"\nWakeword model training completed. Model saved to {model_path}")



if __name__ == "__main__":
    # Optimize memory settings
    optimize_memory_settings()

    # Run the main function
    main()