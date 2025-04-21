"""
Audio processing utilities for wake word detection.
"""
import os
import random

import numpy as np
import librosa
import sounddevice as sd
from config import WINDOW_SIZE_SAMPLES, SAMPLE_RATE, INPUT_FEATURES, WINDOW_SIZE_MS, PREEMPHASIS_COEFF, FFT_SIZE, \
    HOP_LENGTH_MS
import soundfile as sf

def extract_mfcc_features(audio, sample_rate=SAMPLE_RATE, n_mfcc=INPUT_FEATURES):
    """
    Extract MFCC features optimized for ESP32-S3 processing.
    Includes additional robustness techniques for real-world conditions.

    Args:
        audio: Audio signal array
        sample_rate: Audio sample rate (default: 16000)
        n_mfcc: Number of MFCC features to extract (default: 13)

    Returns:
        numpy array of MFCC features
    """
    # Ensure audio is numpy array
    audio = np.array(audio, dtype=np.float32)

    # Ensure audio is not empty
    if len(audio) == 0:
        # Return a zero array with expected dimensions
        return np.zeros((int(WINDOW_SIZE_MS / 10), n_mfcc))

    # Apply pre-emphasis to enhance high frequencies (improves speech recognition)
    preemphasis_coeff = PREEMPHASIS_COEFF
    emphasized_audio = np.append(audio[0], audio[1:] - preemphasis_coeff * audio[:-1])

    try:
        # Extract MFCCs with settings optimized for wake word detection
        mfccs = librosa.feature.mfcc(
            y=emphasized_audio,
            sr=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=FFT_SIZE,  # Reduced FFT size for efficiency
            hop_length=160,  # 10ms hop for features
            win_length=400,  # 25ms window
            window='hamming'  # Hamming window for better spectral characteristics
        )

        # Transpose to time-first dimensions
        features = mfccs.T

        # Normalize features for better training stability
        # Use per-feature normalization to preserve relative importance
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-8
        features = (features - mean) / std

        # Pad or trim to ensure consistent size (for fixed-length input to model)
        target_length = int(WINDOW_SIZE_MS / 10)  # 50 frames for 500ms window

        if features.shape[0] < target_length:
            # Pad with zeros if shorter
            padding = np.zeros((target_length - features.shape[0], features.shape[1]))
            features = np.vstack([features, padding])
        elif features.shape[0] > target_length:
            # Trim if longer
            features = features[:target_length, :]

        return features

    except Exception as e:
        print(f"Error extracting MFCC features: {e}")
        # Return empty feature set with correct dimensions in case of error
        return np.zeros((int(WINDOW_SIZE_MS / 10), n_mfcc))


class AudioBuffer:
    """
    Circular buffer for continuous audio processing.
    Captures audio from microphone and maintains a sliding window.
    """

    def __init__(self, buffer_duration_ms=1000):
        """
        Initialize audio buffer.

        Args:
            buffer_duration_ms: Total buffer duration in milliseconds
        """
        self.sample_rate = SAMPLE_RATE
        self.buffer_size = int(buffer_duration_ms * self.sample_rate / 1000)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.is_recording = False
        self.stream = None

    def audio_callback(self, indata, frames, time, status):
        """
        Callback for audio stream processing.

        Args:
            indata: Input audio data
            frames: Number of frames
            time: Time info
            status: Status flags
        """
        if status:
            print(f"Audio callback status: {status}")

        # Convert to mono if stereo
        if indata.shape[1] > 1:
            audio_mono = np.mean(indata, axis=1)
        else:
            audio_mono = indata[:, 0]

        # Update buffer (shift old samples out, add new samples)
        self.buffer = np.roll(self.buffer, -len(audio_mono))
        self.buffer[-len(audio_mono):] = audio_mono

    def start_recording(self):
        """Start recording from the microphone."""
        if self.is_recording:
            return

        try:
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=int(HOP_LENGTH_MS * self.sample_rate / 1000)
            )
            self.stream.start()
            self.is_recording = True
            print("Recording started")
        except Exception as e:
            print(f"Error starting audio stream: {e}")

    def stop_recording(self):
        """Stop recording from the microphone."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.is_recording = False
            print("Recording stopped")

    def get_window(self):
        """
        Get the latest audio window for processing.

        Returns:
            numpy array of latest audio window
        """
        window_size = WINDOW_SIZE_SAMPLES
        return self.buffer[-window_size:].copy()


def save_audio_examples(positives, negatives, sample_rate, num_examples=10, output_dir="audio_examples"):
    """
    Save audio examples from X_train based on y_train labels.

    Args:
        sample_rate: Sample rate of audio in Hz
        num_examples: Number of examples to save for each class (positive/negative)
        output_dir: Directory to save the audio examples
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)


    # Pick random examples
    if len(positives) >= num_examples:
        pos_to_save = random.sample(list(positives), num_examples)
    else:
        pos_to_save = positives

    if len(negatives) >= num_examples:
        neg_to_save = random.sample(list(negatives), num_examples)
    else:
        neg_to_save = negatives

    # Save positive examples
    for i, audio in enumerate(pos_to_save):
        # Save to file
        sf.write(f"{output_dir}/positive_{i + 1}.wav", audio, sample_rate)

    # Save negative examples
    for i, audio in enumerate(neg_to_save):
        # Save to file
        sf.write(f"{output_dir}/negative_{i + 1}.wav", audio, sample_rate)

def save_segment(audio, sample_rate, start_time, end_time, file_id, segment_idx, label,
                 output_dir="wakeword_segments"):
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