import os
import re
import json
import time
from typing import List, Dict, Tuple, Optional, Union, Any

import numpy as np
import librosa
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import random

from config import WAKEWORD, NO_CACHE, SAMPLE_RATE, SAMPLES


class WakewordDataLoader:
    """Efficient data loader for wakeword model training that minimizes memory usage."""

    def __init__(self, base_path: str, wakeword: str, no_cache: bool = NO_CACHE):
        """
        Initialize the wakeword data loader.

        Args:
            base_path: Path to the MLS dataset
            wakeword: The wakeword to search for
            cache_dir: Directory for cache files
            cache_file: File to store wakeword locations
            no_cache: Whether to ignore existing cache
        """
        self.base_path = base_path
        self.wakeword = wakeword.lower()
        self.cache_file = f"cache/wakeword_locations-{self.base_path.replace('/', '-')}.json"
        self.no_cache = no_cache

        # Compiled regex pattern for faster matching
        self.wakeword_pattern = re.compile(r'\b' + re.escape(self.wakeword) + r'\b')


        # Dictionary to store file metadata (without loading audio)
        self.file_metadata = {}

        # Dictionary to store wakeword locations for files that have them
        self.wakeword_locations = {}

        # Load metadata and filter files
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load metadata from transcripts and filter for potential wakeword matches."""
        metadata_path = os.path.join(self.base_path, "transcripts.txt")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        print(f"Scanning transcripts for potential wakeword matches: '{self.wakeword}'")

        # First pass: scan all transcripts and build metadata dictionary
        start_time = time.time()
        potential_matches = 0
        total_files = 0

        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_files += 1
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    file_id = parts[0]
                    transcript = parts[1].lower()  # Convert to lowercase for case-insensitive matching

                    # Check if transcript contains the wakeword
                    has_wakeword = bool(self.wakeword_pattern.search(transcript))

                    if has_wakeword:
                        potential_matches += 1

                    # Construct the path without loading the file
                    split_id = file_id.split('_')
                    speaker_id = split_id[0]
                    book_id = split_id[1]
                    audio_path = os.path.join(self.base_path, "audio", speaker_id, book_id, f"{file_id}.opus")

                    # Store metadata
                    self.file_metadata[file_id] = {
                        "path": audio_path,
                        "sentence": transcript,
                        "has_wakeword": has_wakeword
                    }

        elapsed = time.time() - start_time
        print(f"Scanned {total_files} transcripts in {elapsed:.2f} seconds")
        print(f"Found {potential_matches} potential matches containing the wakeword '{self.wakeword}'")
        print(f"Filtering ratio: {potential_matches / total_files:.2%}")

        # Try to load wakeword locations from cache
        self._load_wakeword_locations()

    def _load_wakeword_locations(self) -> None:
        """Load wakeword locations from cache if available."""
        if os.path.exists(self.cache_file) and not self.no_cache:
            print(f"Loading wakeword locations from cache: {self.cache_file}")
            with open(self.cache_file, 'r') as f:
                self.wakeword_locations = json.load(f)

                # Count files with confirmed wakewords
                confirmed = sum(1 for locs in self.wakeword_locations.values() if locs)
                print(f"Cache contains confirmed wakeword locations for {confirmed} files")
        else:
            print("No cache found or cache disabled. Wakeword locations will be determined during processing.")

    def save_wakeword_locations(self) -> None:
        """Save the current wakeword locations to cache."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.wakeword_locations, f)
        print(f"Saved wakeword locations to cache: {self.cache_file}")

    def get_filtered_dataset(self, include_all: bool = False) -> Dataset:
        """
        Get a filtered HuggingFace dataset with only relevant files.

        Args:
            include_all: Whether to include all files (for negative examples)

        Returns:
            HuggingFace Dataset object with filtered files
        """
        # Select files based on filtering criteria
        selected_files = []

        # Include files with potential wakeword matches
        wakeword_files = [
            file_id for file_id, meta in self.file_metadata.items()
            if meta["has_wakeword"]
        ]

        # If we need to include other files (for negative examples)
        if include_all:
            # Include some files without wakeword for balance
            non_wakeword_files = [
                file_id for file_id, meta in self.file_metadata.items()
                if not meta["has_wakeword"]
            ]

            # Determine how many non-wakeword files to include
            # (3x the number of wakeword files, but not more than what's available)
            num_non_wakeword = min(len(non_wakeword_files), len(wakeword_files) * 3)

            # Randomly sample from non-wakeword files
            sampled_non_wakeword = random.sample(non_wakeword_files, num_non_wakeword)

            # Combine both lists
            selected_file_ids = wakeword_files + sampled_non_wakeword
            random.shuffle(selected_file_ids)  # Shuffle for balanced batches
        else:
            # Only use files with wakeword
            selected_file_ids = wakeword_files

        # Apply max_files limit if specified
        if SAMPLES is not None:
            selected_file_ids = selected_file_ids[:SAMPLES]

        # Build dataset entries
        for file_id in selected_file_ids:
            meta = self.file_metadata[file_id]
            selected_files.append({
                "file_id": file_id,
                "path": meta["path"],
                "sentence": meta["sentence"],
                "has_wakeword": meta["has_wakeword"]
            })

        print(f"Selected {len(selected_files)} files out of {len(self.file_metadata)} total files")
        print(f"  - Files with potential wakeword: {len(wakeword_files)}")
        if include_all:
            print(f"  - Files without wakeword (for negative examples): {len(selected_files) - len(wakeword_files)}")

        # Create DataFrame and convert to Dataset
        df = pd.DataFrame(selected_files)
        dataset = Dataset.from_pandas(df)

        return dataset

    def lazy_load_audio(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lazily load audio for a batch.
        This function is meant to be used with dataset.map().

        Args:
            batch: Dataset batch with file paths

        Returns:
            Batch with audio loaded
        """
        try:
            # Load audio only when needed
            speech_array, sampling_rate = librosa.load(batch["path"], sr=SAMPLE_RATE)
            batch["speech"] = speech_array
            batch["sampling_rate"] = sampling_rate
        except Exception as e:
            print(f"Error loading audio file {batch['path']}: {e}")
            # Use a placeholder to avoid breaking the dataset
            batch["speech"] = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 second of silence
            batch["sampling_rate"] = SAMPLE_RATE

        return batch

    def get_wakeword_locations(self, file_id: str) -> List[Tuple[float, float]]:
        """
        Get wakeword locations for a specific file.

        Args:
            file_id: File identifier

        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        # Return empty list if file doesn't exist
        if file_id not in self.file_metadata:
            return []

        # Return from cache if available
        if file_id in self.wakeword_locations:
            return self.wakeword_locations[file_id]

        # If not in cache, return empty list (should be populated by find_wakeword function)
        return []

    def update_wakeword_location(self, file_id: str, locations: List[Tuple[float, float]]) -> None:
        """
        Update wakeword locations for a specific file.

        Args:
            file_id: File identifier
            locations: List of (start_time, end_time) tuples in seconds
        """
        self.wakeword_locations[file_id] = locations

    def get_potential_wakeword_files(self) -> List[str]:
        """
        Get list of file IDs that potentially contain the wakeword.

        Returns:
            List of file IDs
        """
        return [
            file_id for file_id, meta in self.file_metadata.items()
            if meta["has_wakeword"]
        ]

    def get_file_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific file.

        Args:
            file_id: File identifier

        Returns:
            Metadata dictionary or None if file doesn't exist
        """
        return self.file_metadata.get(file_id)


def load_dataset_with_efficient_filtering(base_path: str, max_files: Optional[int] = None,
                                          include_negatives: bool = True) -> Tuple[
    Dataset, Dict[str, List[Tuple[float, float]]]]:
    """
    Load and filter the dataset efficiently.

    Args:
        base_path: Path to the MLS dataset
        max_files: Maximum number of files to include
        include_negatives: Whether to include negative examples (files without wakeword)

    Returns:
        Tuple of (filtered_dataset, wakeword_locations)
    """
    # Initialize the data loader
    data_loader = WakewordDataLoader(base_path, WAKEWORD)

    # Get filtered dataset
    filtered_dataset = data_loader.get_filtered_dataset(
        include_all=include_negatives,
        max_files=max_files
    )

    # Lazily load audio data
    dataset_with_audio = filtered_dataset.map(
        data_loader.lazy_load_audio,
        num_proc=4,  # Adjust based on available CPU cores
        desc="Loading audio files"
    )

    # Get wakeword locations
    wakeword_locations = {}
    for file_id in tqdm(filtered_dataset["file_id"], desc="Getting wakeword locations"):
        wakeword_locations[file_id] = data_loader.get_wakeword_locations(file_id)

    return dataset_with_audio, wakeword_locations


def load_audio(path: str, sample_rate: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Load audio from file with error handling.

    Args:
        path: Path to audio file
        sample_rate: Target sample rate

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        speech_array, sampling_rate = librosa.load(path, sr=sample_rate)
        return speech_array, sampling_rate
    except Exception as e:
        print(f"Error loading audio file {path}: {e}")
        # Return 1 second of silence as fallback
        return np.zeros(sample_rate, dtype=np.float32), sample_rate


if __name__ == "__main__":
    # Example usage
    MLS_PATH = "/home/js/Downloads/mls_german_opus/train"
    MAX_FILES = 5000  # Limit for testing

    # Load dataset with efficient filtering
    dataset, wakeword_locations = load_dataset_with_efficient_filtering(
        MLS_PATH,
        max_files=MAX_FILES,
        include_negatives=True
    )

    # Print statistics
    positive_count = sum(1 for locs in wakeword_locations.values() if locs)
    print(f"Dataset loaded with {len(dataset)} files")
    print(f"Files with confirmed wakeword locations: {positive_count}")