import random
import re
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm

from audio_utils import extract_mfcc_features, save_segment
from config import SAMPLE_RATE, INPUT_FEATURES, WINDOW_SIZE_MS, WAKEWORD, WINDOW_SIZE_SAMPLES
from dataloader import WakewordDataLoader, load_audio
from teacher import run_inference


def prepare_inputs(processor, dataset, device):
    """
    Prepare inputs for the model.

    Args:
        processor: Wav2Vec2Processor
        dataset: Dataset with speech samples
        device: Torch device

    Returns:
        Inputs dictionary ready for the model
    """
    # Process audio samples - first on CPU to get proper shape
    inputs = processor(
        dataset["speech"],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )

    # Then move to GPU with correct dtype
    inputs = {
        'input_values': inputs.input_values.to(device, dtype=torch.float16),
        'attention_mask': inputs.attention_mask.to(device) if hasattr(inputs, 'attention_mask') and
                                                       inputs.attention_mask is not None else None
    }

    return inputs


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





def create_training_data(dataset, wakeword_locations):
    """
    Create training data for wake word detection model using sliding window approach.
    Generates fixed-size windows over all files with generous overlap.
    Windows that completely contain a wakeword are labeled as positive.

    Args:
        dataset: Dataset containing audio samples
        wakeword_locations: Dictionary of wake word timestamps

    Returns:
        X_train: Training features
        y_train: Training labels
    """


    positive_examples = []
    negative_examples = []

    # positive_raw = []
    # negative_raw = []

    print("Creating training data with sliding window approach...")

    # Define sliding window parameters
    overlap_ratio = 0.75  # 75% overlap between consecutive windows
    hop_length = int(WINDOW_SIZE_SAMPLES * (1 - overlap_ratio))  # Hop length based on overlap

    # Initialize counters
    total_positive = 0
    total_negative = 0
    total_windows = 0

    # Process all files
    file_indices = list(range(len(dataset)))
    total_audio_duration = 0

    # First count total audio duration for progress estimation
    for idx in tqdm(file_indices, desc="Calculating total audio duration"):
        audio = dataset[idx]["speech"]
        total_audio_duration += len(audio) / SAMPLE_RATE

    # Create master progress bar
    master_pbar = tqdm(total=int(total_audio_duration), desc="Processing audio", unit="sec")

    for idx in file_indices:
        file_id = dataset[idx]["file_id"]
        audio = dataset[idx]["speech"]
        file_duration = len(audio) / SAMPLE_RATE

        # Skip files with no audio or too short
        if len(audio) < WINDOW_SIZE_SAMPLES:
            master_pbar.update(int(file_duration))
            continue

        # Get wake word locations for this file
        locations = wakeword_locations.get(file_id, [])

        # Convert audio to numpy array if it's not already
        audio = np.array(audio, dtype=np.float32)

        # Generate sliding windows
        file_windows = 0
        file_positives = 0
        file_negatives = 0

        for start_idx in range(0, len(audio) - WINDOW_SIZE_SAMPLES + 1, hop_length):
            end_idx = start_idx + WINDOW_SIZE_SAMPLES
            window = audio[start_idx:end_idx]

            # Convert window position to time
            window_start_time = start_idx / SAMPLE_RATE
            window_end_time = end_idx / SAMPLE_RATE

            # Determine if window contains wakeword
            is_positive = False
            for loc_start, loc_end in locations:
                # Check if wakeword is completely contained within the window
                if window_start_time <= loc_start and window_end_time >= loc_end:
                    is_positive = True
                    break
            # if random.random() < 0.01:
            #     if is_positive:
            #         positive_raw.append(window)
            #     else:
            #         negative_raw.append(window)
            try:
                # Extract MFCC features
                mfcc_features = extract_mfcc_features(window)

                # Store example based on label
                if is_positive:
                    positive_examples.append((mfcc_features, 1.0))
                    file_positives += 1
                    total_positive += 1
                else:
                    # Store all negative examples, we'll balance later
                    negative_examples.append((mfcc_features, 0.0))
                    file_negatives += 1
                    total_negative += 1

                file_windows += 1
                total_windows += 1

                # Print intermediate statistics for large files
                if file_windows % 1000 == 0:
                    print(f"  File {file_id}: Processed {file_windows} windows, "
                          f"Positive: {file_positives}, Negative: {file_negatives}")

            except Exception as e:
                print(f"Error processing window at position {start_idx} in file {file_id}: {e}")

        # Update master progress bar
        master_pbar.update(int(file_duration))
        master_pbar.set_postfix({
            'pos': total_positive,
            'neg': total_negative,
            'windows': total_windows
        })

    master_pbar.close()
    print(f"Processed {len(file_indices)} files, generated {total_windows} windows")
    print(f"Raw counts - Positive: {total_positive}, Negative: {total_negative}")

    # save_audio_examples(positive_raw, negative_raw, SAMPLE_RATE, 10)

    # Balance the dataset - we want to keep a reasonable ratio
    max_negative = min(len(negative_examples), 3 * len(positive_examples))

    # Check if we have enough examples
    if not positive_examples:
        print("No positive examples collected! Check wakeword presence in dataset.")
        # Return empty arrays with proper shape
        return np.zeros((0, int(WINDOW_SIZE_MS / 10), INPUT_FEATURES)), np.zeros(0)

    # If we have too few positives, use data augmentation
    if 100 > len(positive_examples) > 0:
        print(f"Only {len(positive_examples)} positive examples found. Adding augmentations...")

        # Create augmented examples
        augmented_positives = []
        for features, label in tqdm(positive_examples, desc="Augmenting positive examples"):
            # Original example
            augmented_positives.append((features, label))

            # Add noise (SNR variation)
            noise_scale = 0.1
            noisy_features = features + noise_scale * np.random.randn(*features.shape)
            augmented_positives.append((noisy_features, label))

            # Time stretching simulation (feature warping)
            stretch_factor = 0.95
            time_steps = features.shape[0]
            stretched_indices = np.clip(
                np.round(np.linspace(0, time_steps - 1, int(time_steps * stretch_factor))).astype(int),
                0, time_steps - 1
            )
            stretched_features = features[stretched_indices, :]
            # Resize back to original shape
            if stretched_features.shape[0] < features.shape[0]:
                padding = np.zeros((features.shape[0] - stretched_features.shape[0], features.shape[1]))
                stretched_features = np.vstack([stretched_features, padding])
            augmented_positives.append((stretched_features, label))

            # Frequency masking (mask some frequency bands)
            freq_mask = features.copy()
            mask_size = max(1, features.shape[1] // 4)
            mask_start = np.random.randint(0, features.shape[1] - mask_size)
            freq_mask[:, mask_start:mask_start + mask_size] = 0
            augmented_positives.append((freq_mask, label))

        positive_examples = augmented_positives
        print(f"After augmentation: {len(positive_examples)} positive examples")

    # If we have too few negatives, just take random segments from the audio
    if len(negative_examples) < len(positive_examples):
        print("Not enough negative examples, generating random ones...")
        for _ in range(3 * len(positive_examples) - len(negative_examples)):
            random_features = np.random.randn(int(WINDOW_SIZE_MS / 10), INPUT_FEATURES)
            negative_examples.append((random_features, 0.0))

    # Combine balanced examples
    print(f"Balancing dataset - keeping {len(positive_examples)} positive and {max_negative} negative examples")

    # Shuffle negatives before selecting subset
    np.random.shuffle(negative_examples)
    combined_examples = positive_examples + negative_examples[:max_negative]

    # Shuffle the dataset
    np.random.shuffle(combined_examples)

    # Unpack features and labels
    X_train, y_train = map(list, zip(*combined_examples))

    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"Final training dataset: {len(X_train)} examples")
    print(f"  Positive examples: {sum(y_train == 1.0)}")
    print(f"  Negative examples: {sum(y_train == 0.0)}")
    print(f"  Positive/Negative ratio: {sum(y_train == 1.0) / max(1, sum(y_train == 0.0)):.2f}")


    return X_train, y_train


class WakewordProcessor:
    """
    Wakeword processing system that uses memory-efficient loading
    and processes only the files that are likely to contain wakewords.
    """

    def __init__(self, base_path: str, teacher_model, teacher_processor, device):
        """
        Initialize the wakeword processor.

        Args:
            base_path: Path to the MLS dataset
            teacher_model: The wav2vec2 teacher model
            teacher_processor: The wav2vec2 processor
            device: Torch device
        """
        self.base_path = base_path
        self.teacher_model = teacher_model
        self.teacher_processor = teacher_processor
        self.device = device

        # Create data loader
        self.data_loader = WakewordDataLoader(base_path, WAKEWORD)

        # Load or create wakeword locations
        self.wakeword_locations = self._load_or_create_wakeword_cache()

    def _load_or_create_wakeword_cache(self) -> Dict[str, List[Tuple[float, float]]]:
        """
        Load wakeword locations from cache or create cache using teacher model with binary search.
        Only process files that potentially have the wakeword based on transcript.

        Returns:
            Dictionary mapping file_id to wake word locations
        """
        if self.data_loader.wakeword_locations:
            return self.data_loader.wakeword_locations

        print("Generating wake word locations using teacher model with binary search...")
        wakeword_locations = {}

        # Get list of files that potentially have the wakeword
        potential_files = self.data_loader.get_potential_wakeword_files()
        print(f"Processing {len(potential_files)} files that potentially contain the wakeword")

        # Initialize with empty lists for files that don't potentially have the wakeword
        for file_id in self.data_loader.file_metadata:
            if file_id not in potential_files:
                wakeword_locations[file_id] = []

        # Process files with potential wakewords
        for file_id in tqdm(potential_files, desc="Processing potential wakeword files"):
            meta = self.data_loader.get_file_metadata(file_id)
            if not meta:
                wakeword_locations[file_id] = []
                continue

            sentence = meta["sentence"]
            print(f"Processing file {file_id}: {sentence}")

            try:
                # Load audio for this file
                speech_array, sampling_rate = load_audio(meta["path"])

                # Use binary search approach to find wake word
                word_timestamps = find_wakeword_with_binary_search(
                    speech_array,
                    sampling_rate,
                    self.teacher_model,
                    self.teacher_processor,
                    WAKEWORD.strip(),
                    self.device,
                    file_id,
                    "wakeword_segments",
                    sentence
                )

                wakeword_locations[file_id] = word_timestamps

                # Update the data loader's cache as we go
                self.data_loader.update_wakeword_location(file_id, word_timestamps)

            except Exception as e:
                print(f"Error processing file {file_id}: {e}")
                wakeword_locations[file_id] = []

        # Save to cache
        self.data_loader.save_wakeword_locations()

        return wakeword_locations

    def process_dataset(self, include_negatives: bool = True) -> Tuple[Any, Dict]:
        """
        Process the dataset to extract wakeword segments and create training data.

        Args:
            include_negatives: Whether to include negative examples

        Returns:
            Tuple of (dataset_with_audio, wakeword_locations)
        """
        # Get filtered dataset
        filtered_dataset = self.data_loader.get_filtered_dataset(
            include_all=include_negatives,
        )

        # Lazily load audio data - this will only happen when the item is accessed
        dataset_with_audio = filtered_dataset.map(
            self.data_loader.lazy_load_audio,
            num_proc=1,  # Careful with multiprocessing when loading large files
            desc="Preparing dataset for lazy loading"
        )

        print(f"Dataset prepared with {len(dataset_with_audio)} files")
        print(f"  - Files with potential wakeword: {sum(1 for x in dataset_with_audio if x.get('has_wakeword', False))}")

        return dataset_with_audio, self.wakeword_locations