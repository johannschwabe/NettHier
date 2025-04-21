"""
Real-time wake word detection from microphone input.
"""

import time
import argparse
import sounddevice as sd
from threading import Thread, Event
import config
from audio_utils import AudioBuffer
from model_utils import WakeWordDetector


def list_audio_devices():
    """List available audio input devices."""
    print("\nAvailable audio input devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # Only show input devices
            print(f"{i}: {device['name']} (Inputs: {device['max_input_channels']})")
    print()


def run_detection(args):
    """
    Run wake word detection on microphone input.

    Args:
        args: Command line arguments
    """
    # Print system information
    print(f"Wake Word Detection System")
    print(f"Wake Word: '{config.WAKEWORD}'")
    print(f"Model: {config.MODEL_PATH}")
    print(f"Sample Rate: {config.SAMPLE_RATE} Hz")
    print(f"Window Size: {config.WINDOW_SIZE_MS} ms")
    print(f"Hop Length: {config.HOP_LENGTH_MS} ms")

    if args.list_devices:
        list_audio_devices()
        return

    if args.device is not None:
        try:
            sd.query_devices(args.device)  # Check if device exists
            print(f"Using audio device: {sd.query_devices(args.device)['name']}")
        except Exception as e:
            print(f"Error selecting audio device {args.device}: {e}")
            list_audio_devices()
            return

    try:
        # Create wake word detector
        detector = WakeWordDetector(config.MODEL_PATH)

        # Create audio buffer for continuous recording
        audio_buffer = AudioBuffer(buffer_duration_ms=1500)  # Buffer size larger than window

        # Start recording
        audio_buffer.start_recording()

        # Create a stopping event
        stop_event = Event()

        def detection_loop():
            """Run detection in a loop."""
            print("\nListening for wake word. Press Ctrl+C to stop...")
            while not stop_event.is_set():
                # Get latest audio window
                audio_window = audio_buffer.get_window()

                # Process audio for wake word
                detector.process_audio(audio_window)


                # Sleep to avoid excessive CPU usage
                time.sleep(config.HOP_LENGTH_MS / 2000)

        # Start detection in separate thread
        detection_thread = Thread(target=detection_loop)
        detection_thread.start()

        # Wait for keyboard interrupt
        try:
            while detection_thread.is_alive():
                detection_thread.join(1)
        except KeyboardInterrupt:
            print("\nStopping...")
            stop_event.set()

        # Clean up
        audio_buffer.stop_recording()

    except Exception as e:
        print(f"Error running wake word detection: {e}")


def main():
    """Main function to parse arguments and run detection."""
    parser = argparse.ArgumentParser(description="Wake Word Detection System")
    parser.add_argument("--device", type=int, help="Audio input device index")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices")
    parser.add_argument("--threshold", type=float, help=f"Detection threshold (default: {config.DETECTION_THRESHOLD})")

    args = parser.parse_args()

    # Override config values if specified
    if args.threshold:
        config.DETECTION_THRESHOLD = args.threshold

    run_detection(args)


if __name__ == "__main__":
    main()