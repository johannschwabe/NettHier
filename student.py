import multiprocessing
import os
import json
import re
import time
from functools import partial

import numpy as np
import torch
from tqdm import tqdm
import soundfile as sf

import concurrent.futures

from audio_utils import extract_mfcc_features
from config import SAMPLE_RATE, INPUT_FEATURES, WINDOW_SIZE_MS, CACHE_DIR, NO_CACHE, CACHE_FILE, WAKEWORD, \
    WINDOW_SIZE_SAMPLES, HOP_LENGTH_SAMPLES, HIDDEN_SIZE, NUM_LAYERS, HOP_LENGTH_MS, SAMPLES
from model_utils import TinyWakeWordModel
from teacher import load_mls_dataset, speech_file_to_array_fn, check_device, prepare_inputs, run_inference
from teacher import load_model_and_processor



def find_wakeword_with_binary_search(audio, sample_rate, teacher_model, teacher_processor, wakeword, device,
                                     file_id=None, output_dir="wakeword_segments", sentence=None):
    """
    Find wake word using binary search approach to speed up detection.

    Args:
        audio: Audio signal array
        sample_rate: Audio sample rate in Hz
        teacher_model: The wav2vec2 teacher model
        teacher_processor: The wav2vec2 processor
        wakeword: Wake word to search for
        device: Torch device
        file_id: Identifier for the source audio file
        output_dir: Directory to save audio segments
        sentence: Transcript of the audio to count expected occurrences

    Returns:
        List of (start_time, end_time) tuples in seconds
    """

    # Ensure audio is a numpy array
    audio = np.array(audio, dtype=np.float32)

    # Add 1 second of silence padding at beginning and end
    padding_size = sample_rate  # 1 second of silence
    padded_audio = np.pad(audio, (padding_size, padding_size), 'constant', constant_values=0)

    # Keep track of original audio position
    original_start = padding_size

    # Count expected occurrences based on transcript if provided
    expected_occurrences = 0
    if sentence:
        # Count case-insensitive occurrences in transcript
        expected_occurrences = len(re.findall(wakeword, sentence.lower()))
        print(f"Expected {expected_occurrences} occurrences of '{wakeword}' based on transcript")

    # Initialize list to store found wake word segments
    wakeword_segments = []

    # Define minimum segment size that could contain the wake word (in seconds)
    # This is a tunable parameter - start with a conservative estimate
    min_segment_duration = 0.5  # seconds
    min_segment_size = int(min_segment_duration * sample_rate)

    # Define overlap to ensure wake word isn't missed at boundaries
    # The overlap should be longer than the expected duration of the wake word
    overlap_duration = 0.3  # seconds
    overlap_size = int(overlap_duration * sample_rate)

    # Queue to store segments for processing
    segments_to_process = [(0, len(padded_audio))]

    # Counter for processed segments
    processed_segments = 0

    # Process segments until we find all expected occurrences or exhaust search
    while segments_to_process and (not expected_occurrences or len(wakeword_segments) < expected_occurrences):
        start, end = segments_to_process.pop(0)
        segment_size = end - start
        processed_segments += 1

        # If the segment is too small, skip it
        if segment_size < min_segment_size:
            continue

        # Process the current segment with the teacher model
        segment_audio = padded_audio[start:end]

        # Create a mock dataset with the segment audio
        # The prepare_inputs function expects a dataset with a 'speech' key
        mock_dataset = {"speech": segment_audio}

        # Prepare inputs using the helper function
        inputs = prepare_inputs(teacher_processor, mock_dataset, device)

        # Run inference using the helper function
        transcriptions = run_inference(teacher_model, inputs, teacher_processor)
        transcription = transcriptions[0].lower()  # Get the first (and only) transcription

        # Check if wake word is in this segment
        if wakeword in transcription:
            # If the segment is small enough, consider it a match
            if segment_size <= min_segment_size * 4:
                # Time relative to padded audio
                padded_start_time = start / sample_rate
                padded_end_time = end / sample_rate
                print(
                    f"Found wake word in segment ({padded_start_time:.2f}s - {padded_end_time:.2f}s): {transcription}")

                # Adjust to original audio coordinates (remove padding offset)
                original_start_time = max(0, (start - original_start) / sample_rate)
                original_end_time = min(len(audio) / sample_rate, (end - original_start) / sample_rate)

                # Check if this segment overlaps with previously found segments
                is_overlapping = False
                for seg_start, seg_end in wakeword_segments:
                    if not (original_end_time < seg_start or original_start_time > seg_end):
                        is_overlapping = True
                        break

                if not is_overlapping:
                    wakeword_segments.append((original_start_time, original_end_time))

                    # Save audio segment if requested
                    if file_id:
                        save_segment(audio, sample_rate, original_start_time, original_end_time,
                                     file_id, len(wakeword_segments) - 1, "binary_search", output_dir)

            # If the segment is large, split it and add to queue
            else:
                mid = (start + end) // 2

                # Add both halves to the queue with overlap
                # First half: from start to mid + overlap
                segments_to_process.append((start, min(end, mid + overlap_size)))
                # Second half: from mid - overlap to end
                segments_to_process.append((max(start, mid - overlap_size), end))


    print(f"Binary search processed {processed_segments} segments")
    print(f"Found {len(wakeword_segments)} wake word occurrences")

    # Sort segments by start time
    wakeword_segments.sort(key=lambda x: x[0])

    # Refine the segments if needed
    refined_segments = refine_wakeword_segments(audio, sample_rate, wakeword_segments, teacher_model,
                                                teacher_processor, wakeword, device, file_id, output_dir)

    return refined_segments


def refine_wakeword_segments(audio, sample_rate, segments, teacher_model, teacher_processor, wakeword, device,
                             file_id=None, output_dir="wakeword_segments"):
    """
    Refine wake word segments by progressively trimming from start and end until
    the wake word is no longer detected, then reducing the trim rate.

    Args:
        audio: Audio signal array
        sample_rate: Audio sample rate in Hz
        segments: List of (start_time, end_time) tuples in seconds
        teacher_model: The wav2vec2 teacher model
        teacher_processor: The wav2vec2 processor
        wakeword: Wake word to search for
        device: Torch device
        file_id: Identifier for the source audio file
        output_dir: Directory to save audio segments

    Returns:
        List of refined (start_time, end_time) tuples in seconds
    """

    refined_segments = []

    for seg_idx, (start_time, end_time) in enumerate(segments):
        # Convert times to sample indices
        start_sample = max(0, int(start_time * sample_rate))
        end_sample = min(len(audio), int(end_time * sample_rate))
        original_start = start_sample
        original_end = end_sample

        # Set initial trim rates in samples
        # Start with larger steps (0.1s worth of samples) and reduce as needed
        trim_rate_start = int(0.2 * sample_rate)
        trim_rate_end = int(0.2 * sample_rate)

        # Set minimum trim rate (about 10ms worth of samples)
        min_trim_rate = int(0.05 * sample_rate)

        # Refine the start boundary
        while trim_rate_start >= min_trim_rate:
            # Make a copy of the current start position before trimming
            prev_start = start_sample

            # Try trimming from the start
            test_start = start_sample + trim_rate_start
            if end_sample - test_start <= start_sample * 0.1:
                trim_rate_start = trim_rate_start // 2
                continue

            # Extract segment with the new start position
            segment = audio[test_start:end_sample]

            # Create a mock dataset with the segment audio
            mock_dataset = {"speech": segment}

            # Prepare inputs and run inference
            inputs = prepare_inputs(teacher_processor, mock_dataset, device)
            transcriptions = run_inference(teacher_model, inputs, teacher_processor)
            transcription = transcriptions[0].lower()

            # Check if wake word is still present
            if wakeword in transcription:
                # We can trim more from the start
                start_sample = test_start
            else:
                # We've trimmed too much, go back and reduce trim rate
                start_sample = prev_start
                trim_rate_start = trim_rate_start // 2

        found_start = start_sample
        start_sample -= int(sample_rate * 0.5)
        # Refine the end boundary
        while trim_rate_end >= min_trim_rate:
            # Make a copy of the current end position before trimming
            prev_end = end_sample

            # Try trimming from the end
            test_end = end_sample - trim_rate_end
            if test_end - start_sample <= start_sample * 0.1:
                trim_rate_end = trim_rate_end // 2
                continue

            # Extract segment with the new end position
            segment = audio[start_sample:test_end]

            # Create a mock dataset with the segment audio
            mock_dataset = {"speech": segment}

            # Prepare inputs and run inference
            inputs = prepare_inputs(teacher_processor, mock_dataset, device)
            transcriptions = run_inference(teacher_model, inputs, teacher_processor)
            transcription = transcriptions[0].lower()

            # Check if wake word is still present
            if wakeword in transcription:
                # We can trim more from the end
                end_sample = test_end
            else:
                # We've trimmed too much, go back and reduce trim rate
                end_sample = prev_end
                trim_rate_end = trim_rate_end // 2
            print(f"transcription: {transcription}, test_end: {end_sample}, trim_rate_end: {trim_rate_end}")


        # Convert sample indices back to time
        refined_start_time = found_start / sample_rate
        refined_end_time = end_sample / sample_rate + 0.05

        print(f"Segment {seg_idx + 1}: Original ({start_time:.3f}s - {end_time:.3f}s), "
              f"Refined ({refined_start_time:.3f}s - {refined_end_time:.3f}s)")
        print(f"Trimmed {(start_sample - original_start) / sample_rate:.3f}s from start, "
              f"{(original_end - end_sample) / sample_rate:.3f}s from end")

        # Save refined segment if requested
        if file_id:
            save_segment(audio, sample_rate, refined_start_time, refined_end_time,
                         file_id, seg_idx, "refined", output_dir)

        refined_segments.append((refined_start_time, refined_end_time))

    return refined_segments


def prefilter_chunk(indices, dataset, wakeword):
    # Compile pattern here since we can't pass compiled patterns between processes
    wakeword_pattern = re.compile(r'\b' + re.escape(wakeword.lower()) + r'\b')
    results = []
    for idx in indices:
        try:
            file_id = dataset[idx]["file_id"]
            sentence = dataset[idx].get("sentence", "").lower()

            # Use regex for faster matching
            has_wakeword = bool(wakeword_pattern.search(sentence))

            results.append((idx, file_id, has_wakeword))
        except Exception as e:
            # Handle any potential issues with the dataset
            print(f"Error in prefiltering index {idx}: {e}")
            results.append((idx, f"error_{idx}", False))
    return results


def load_or_create_wakeword_cache(dataset, teacher_model, teacher_processor, device):
    """
    Load wake word locations from cache or create cache using teacher model with binary search.
    Uses highly optimized parallel prefiltering.

    Args:
        dataset: Dataset containing audio samples
        teacher_model: The wav2vec2 teacher model
        teacher_processor: The wav2vec2 processor
        device: Torch device

    Returns:
        Dictionary mapping file_id to wake word locations
    """
    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Check if cache file exists
    if os.path.exists(CACHE_FILE) and not NO_CACHE:
        print(f"Loading wake word locations from cache: {CACHE_FILE}")
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)

    print("Generating wake word locations using teacher model with binary search...")
    wakeword_locations = {}

    # Determine optimal chunk size based on CPU count
    # Larger chunks reduce overhead but might distribute work less evenly
    cpu_count = multiprocessing.cpu_count()
    total_items = len(dataset)
    chunk_size = max(100, total_items // (cpu_count * 10))  # Aim for ~10 chunks per CPU

    # Create chunks of indices
    index_chunks = [
        list(range(i, min(i + chunk_size, total_items)))
        for i in range(0, total_items, chunk_size)
    ]

    print(f"Prefiltering dataset in {len(index_chunks)} chunks with {chunk_size} items per chunk...")
    start_time = time.time()

    # Use ProcessPoolExecutor for CPU-bound task
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Create a partial function with the dataset
        chunk_filter_fn = partial(prefilter_chunk, dataset=dataset, wakeword=WAKEWORD)

        # Submit all chunks for processing
        futures = [executor.submit(chunk_filter_fn, chunk) for chunk in index_chunks]

        # Track progress and collect results
        filtered_indices = []
        files_without_wakeword = []

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                           desc="Prefiltering dataset"):
            results = future.result()
            for idx, file_id, has_wakeword in results:
                if has_wakeword:
                    filtered_indices.append(idx)
                else:
                    files_without_wakeword.append(file_id)

    elapsed = time.time() - start_time
    items_per_second = total_items / elapsed
    print(f"Prefiltering completed in {elapsed:.2f} seconds ({items_per_second:.2f} items/sec)")

    # Initialize wakeword_locations with empty lists for files without wake words
    for file_id in files_without_wakeword:
        wakeword_locations[file_id] = []

    print(f"Found {len(filtered_indices)} files containing wake word out of {len(dataset)} total files")

    # Process files with wake words sequentially
    for idx in tqdm(filtered_indices, desc="Processing wake word files"):
        file_id = dataset[idx]["file_id"]
        sentence = dataset[idx]["sentence"]

        print(f"Processing file {file_id}: {sentence}")

        # Get audio signal
        speech = dataset[idx]["speech"]

        try:
            # Use binary search approach to find wake word
            word_timestamps = find_wakeword_with_binary_search(
                speech,
                dataset[idx].get("sampling_rate", SAMPLE_RATE),  # Use provided rate or default
                teacher_model,
                teacher_processor,
                WAKEWORD.strip(),  # Remove any whitespace
                device,
                file_id,
                "wakeword_segments",
                sentence
            )

            wakeword_locations[file_id] = word_timestamps

        except Exception as e:
            print(f"Error processing file {file_id}: {e}")
            wakeword_locations[file_id] = []

    # Save to cache
    with open(CACHE_FILE, 'w') as f:
        json.dump(wakeword_locations, f)

    return wakeword_locations


def save_segment(audio, sample_rate, start_time, end_time, file_id, segment_idx, label, output_dir="wakeword_segments"):
    """
    Save an audio segment to file for validation.

    Args:
        audio: Full audio array
        sample_rate: Audio sample rate
        start_time: Start time in seconds
        end_time: End time in seconds
        file_id: Source file identifier
        segment_idx: Segment index
        label: Additional label for the segment
        output_dir: Output directory
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert times to sample indices
    start_sample = max(0, int(start_time * sample_rate))
    end_sample = min(len(audio), int(end_time * sample_rate))

    # Extract the segment
    segment = audio[start_sample:end_sample]

    # Create filename
    filename = f"{output_dir}/{file_id}_segment{segment_idx}_{label}.wav"

    # Save as WAV file
    sf.write(filename, segment, sample_rate)

    print(f"Saved wake word segment to: {filename}")


def create_training_data(dataset, wakeword_locations):
    """
    Create training data for wake word detection model.
    Balances positive and negative examples for better training.

    Args:
        dataset: Dataset containing audio samples
        wakeword_locations: Dictionary of wake word timestamps

    Returns:
        X_train: Training features
        y_train: Training labels
    """
    positive_examples = []
    negative_examples = []

    print("Creating training data...")
    for i in tqdm(range(len(dataset))):
        file_id = dataset[i]["file_id"]
        audio = dataset[i]["speech"]

        # Convert audio to numpy array if it's not already
        audio = np.array(audio, dtype=np.float32)

        # Get wake word locations for this file
        locations = wakeword_locations[file_id]

        # Skip files with no audio or too short
        if len(audio) < WINDOW_SIZE_SAMPLES:
            continue

        # Process windows
        windows_processed = 0
        positive_windows = 0

        for start_idx in range(0, len(audio) - WINDOW_SIZE_SAMPLES, HOP_LENGTH_SAMPLES):
            # Limit the number of windows per file to prevent memory issues
            if windows_processed > 50:  # Process at most 50 windows per file
                break

            end_idx = start_idx + WINDOW_SIZE_SAMPLES
            window = audio[start_idx:end_idx]

            # Check if window contains wake word
            window_start_time = start_idx / SAMPLE_RATE
            window_end_time = end_idx / SAMPLE_RATE

            # Check for wake word presence
            contains_wakeword = any(
                (loc_start >= window_start_time and loc_start <= window_end_time) or
                (loc_end >= window_start_time and loc_end <= window_end_time) or
                (window_start_time >= loc_start and window_end_time <= loc_end)
                for loc_start, loc_end in locations
            )

            try:
                # Extract MFCC features
                mfcc_features = extract_mfcc_features(window)

                # Store examples
                if contains_wakeword:
                    positive_examples.append((mfcc_features, 1.0))
                    positive_windows += 1

                    # For positive examples, also create slight variations by adding small time shifts
                    # This helps with robustness
                    if len(positive_examples) < 1000:  # Limit augmentations to prevent memory issues
                        # Shift left by 10% of window
                        if start_idx > WINDOW_SIZE_SAMPLES * 0.1:
                            shift_start = int(start_idx - WINDOW_SIZE_SAMPLES * 0.1)
                            shift_end = shift_start + WINDOW_SIZE_SAMPLES
                            if shift_end <= len(audio):
                                shifted_window = audio[shift_start:shift_end]
                                shifted_features = extract_mfcc_features(shifted_window)
                                positive_examples.append((shifted_features, 1.0))

                        # Shift right by 10% of window
                        shift_start = int(start_idx + WINDOW_SIZE_SAMPLES * 0.1)
                        shift_end = shift_start + WINDOW_SIZE_SAMPLES
                        if shift_end <= len(audio):
                            shifted_window = audio[shift_start:shift_end]
                            shifted_features = extract_mfcc_features(shifted_window)
                            positive_examples.append((shifted_features, 1.0))
                else:
                    # Only add a fraction of negative examples to balance dataset
                    # For files with wake word, keep 1:3 ratio of positive:negative
                    if not locations or len(negative_examples) < 3 * len(positive_examples):
                        negative_examples.append((mfcc_features, 0.0))

            except Exception as e:
                print(f"Error processing window in file {file_id}: {e}")
                continue

            windows_processed += 1

            # Early stop if we've found enough positive examples
            if len(positive_examples) >= 5000:
                break

        # Early stop if we've found enough positive examples
        if len(positive_examples) >= 5000:
            print(f"Collected {len(positive_examples)} positive examples - sufficient for training")
            break

    # Combine and balance the dataset
    # Limit negative examples to ensure balanced training
    max_negative = min(len(negative_examples), 3 * len(positive_examples))

    # Check if we have any examples
    if not positive_examples or not negative_examples:
        print("Not enough examples collected. Check wake word presence in dataset.")
        # Return empty arrays with proper shape
        return np.zeros((0, int(WINDOW_SIZE_MS / 10), INPUT_FEATURES)), np.zeros(0)

    combined_examples = positive_examples + negative_examples[:max_negative]

    # Shuffle the dataset
    np.random.shuffle(combined_examples)

    # Unpack features and labels
    X_train, y_train = map(list, zip(*combined_examples))

    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"Created training dataset with {len(X_train)} examples")
    print(f"Positive examples: {sum(y_train == 1.0)}, Negative examples: {sum(y_train == 0.0)}")

    return X_train, y_train


def train_model(X_train, y_train, device, epochs=15, batch_size=64):
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
    model = TinyWakeWordModel().to(device)

    # Use binary cross entropy loss with class weights to handle imbalanced data
    pos_weight = torch.tensor([3.0]).to(device)  # Adjust based on class distribution
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Use AdamW optimizer with learning rate schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Early stopping parameters
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 5
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
        model_path = os.path.join(output_dir, "wakeword_model.pt")
        optimized_model.save(model_path)
        print(f"Quantized model exported to {model_path}")
        model_size = os.path.getsize(model_path) / 1024  # KB

    except Exception as e:
        print(f"Quantization failed: {e}")
        print("Saving non-quantized model instead")
        # If quantization fails, export the regular model
        model_path = os.path.join(output_dir, "wakeword_model.pt")
        torch.save(cpu_model.state_dict(), model_path)
        print(f"Non-quantized model exported to {model_path}")
        model_size = os.path.getsize(model_path) / 1024  # KB

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
        "model_size_bytes": os.path.getsize(model_path),
        "detection_threshold": 0.7
    }

    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Model metadata exported to {metadata_path}")

    # 9. Print model size information
    print(f"Model size: {model_size:.2f} KB")

    return model_path


def main():
    """Main function to train the wake word detection model."""
    # Configuration
    MLS_PATH = "/home/js/Downloads/mls_german_opus/train"  # Update this to your path
    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-german"

    # Check device
    device = check_device()

    # Load teacher model and processor
    teacher_processor, teacher_model = load_model_and_processor(MODEL_ID, device)

    # Load dataset
    print("Loading MLS German dataset...")
    dataset = load_mls_dataset(MLS_PATH, SAMPLES)

    if dataset is None:
        print("Failed to load MLS German dataset")
        return

    # Preprocess dataset
    dataset = dataset.map(speech_file_to_array_fn)

    # Load or create wake word locations cache
    try:
        wakeword_locations = load_or_create_wakeword_cache(
            dataset, teacher_model, teacher_processor, device
        )
    except Exception as e:
        print(f"Error in wake word location extraction: {e}")
        print("Creating an empty cache file for debugging...")
        # Create an empty cache file for debugging
        wakeword_locations = {dataset[i]["file_id"]: [] for i in range(len(dataset))}

        # Add some synthetic wake word locations for testing
        # This allows development to continue even if teacher model extraction fails
        print("Adding synthetic wake word locations for development...")
        import random
        for i in range(min(50, len(dataset))):
            file_id = dataset[i]["file_id"]
            audio = dataset[i]["speech"]
            audio_length = len(audio) / SAMPLE_RATE

            # Create random timestamps for 10% of files
            if random.random() < 0.1 and audio_length > 2.0:
                start_time = random.uniform(0.5, audio_length - 1.5)
                end_time = start_time + random.uniform(0.5, 1.0)
                wakeword_locations[file_id] = [(start_time, end_time)]

        # Save this synthetic cache
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(CACHE_FILE, 'w') as f:
            json.dump(wakeword_locations, f)

    # Create training data
    X_train, y_train = create_training_data(dataset, wakeword_locations)

    # Check if we have enough data for training
    if len(X_train) < 100 or sum(y_train == 1.0) < 10:
        print("Not enough training data, especially positive examples.")
        print(f"Total examples: {len(X_train)}, Positive examples: {sum(y_train == 1.0)}")
        return

    # Train model
    model = train_model(X_train, y_train, device)

    # Export model for ESP32-S3
    model_path = export_for_esp32(model)

    print(f"\nWake word model training completed. Model saved to {model_path}")

    # Evaluate model on a few examples
    print("\nPerforming quick evaluation on sample data...")
    evaluate_on_samples(model, dataset, wakeword_locations, device)


def evaluate_on_samples(model, dataset, wakeword_locations, device, num_samples=5):
    """
    Perform a quick evaluation on a few samples.
    Ensures proper device handling for model and inputs.

    Args:
        model: Trained model
        dataset: Dataset containing audio samples
        wakeword_locations: Dictionary of wake word timestamps
        device: Torch device
        num_samples: Number of samples to evaluate
    """
    # Ensure model is in eval mode and on the correct device
    model.eval()

    # Track which device the model is on - this is critical
    model_device = next(model.parameters()).device
    print(f"Model is on device: {model_device}")

    # Find files with wake words
    files_with_wakeword = [
        i for i in range(len(dataset))
        if wakeword_locations.get(dataset[i]["file_id"], [])
    ]

    # If no files with wake words, select random files
    if not files_with_wakeword:
        files_with_wakeword = list(range(min(num_samples, len(dataset))))

    # Select a subset of files with wake words
    import random
    selected_indices = random.sample(
        files_with_wakeword,
        min(num_samples, len(files_with_wakeword))
    )

    for idx in selected_indices:
        file_id = dataset[idx]["file_id"]
        # Convert audio to numpy array if needed
        audio = np.array(dataset[idx]["speech"], dtype=np.float32)

        locations = wakeword_locations.get(file_id, [])

        print(f"\nEvaluating file: {file_id}")
        print(f"Wake word locations: {locations}")

        # Process the audio in windows
        detections = []

        for start_idx in range(0, len(audio) - WINDOW_SIZE_SAMPLES, HOP_LENGTH_SAMPLES):
            end_idx = start_idx + WINDOW_SIZE_SAMPLES
            window = audio[start_idx:end_idx]

            # Skip if window is too short
            if len(window) < WINDOW_SIZE_SAMPLES:
                continue

            try:
                # Extract features
                features = extract_mfcc_features(window)

                # Convert to tensor and explicitly control device placement
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

                # Make sure input tensor is on same device as model
                features_tensor = features_tensor.to(model_device)

                # Make prediction
                with torch.no_grad():
                    output = model(features_tensor)
                    prediction = torch.sigmoid(output).item()

                window_time = start_idx / SAMPLE_RATE

                # If prediction above threshold, add to detections
                if prediction > 0.7:  # Using 0.7 as threshold for demonstration
                    detections.append((window_time, prediction))

            except Exception as e:
                print(f"Error processing window at {start_idx / SAMPLE_RATE:.2f}s: {e}")
                continue

        # Print detections
        if detections:
            print(f"Detected {len(detections)} potential wake word instances:")
            for time, confidence in detections:
                print(f"  Time: {time:.2f}s, Confidence: {confidence:.4f}")

                # Check if this is close to a known wake word location
                for loc_start, loc_end in locations:
                    if abs(time - loc_start) < 0.5 or abs(time - loc_end) < 0.5:
                        print(f"    ✓ Near known wake word at [{loc_start:.2f}s - {loc_end:.2f}s]")
                        break
                else:
                    if locations:  # Only mark as potential false positive if we know the locations
                        print(f"    ⚠ Potential false positive")
        else:
            print("No wake word detected in this file.")
            if locations:
                print("  ⚠ Missed wake word detection!")


if __name__ == "__main__":
    main()