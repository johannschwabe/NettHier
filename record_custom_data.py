import os
import sounddevice as sd
from datetime import datetime
import soundfile as sf


from config import SAMPLE_RATE, MODEL_ID
from teacher import check_device, load_model_and_processor, prepare_inputs, run_inference



def record_audio(duration, channels=1):
    """
    Record audio from the microphone.

    Args:
        duration: Recording duration in seconds
        channels: Number of audio channels

    Returns:
        Recorded audio as numpy array
    """
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=channels,
        dtype='float32'
    )

    # Wait for the recording to complete
    sd.wait()

    # Convert to mono if needed
    if channels > 1:
        audio_data = audio_data.mean(axis=1)

    print("Recording complete!")
    return audio_data.flatten()


def save_audio(audio_data, file_path):
    """
    Save audio data to WAV file.

    Args:
        audio_data: Audio data as numpy array
        file_path: Output file path
    """

    sf.write(file_path, audio_data, SAMPLE_RATE, format='OGG', subtype="OPUS")

    print(f"Audio saved to {file_path}")




def record_wakeword_samples():
    """
    Record wakeword samples and generate MLS-compatible dataset.
    """


    # Generate a unique session ID based on timestamp
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    speaker_id = "JS"
    duration = 10

    rec_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recorded")
    # Set up output directories
    os.makedirs(rec_dir, exist_ok=True)
    audio_dir = os.path.join(rec_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    session_dir = os.path.join(audio_dir, session_id)
    os.makedirs(session_dir, exist_ok=True)
    speaker_dir = os.path.join(session_dir, speaker_id)
    os.makedirs(speaker_dir, exist_ok=True)


    device = check_device()
    processor, model = load_model_and_processor(device)

    # Record wakeword samples
    recordings = []

    print(f"\n=== Recording Session: {session_id} ===")
    print("Follow the prompts to record each sample")
    iteration = 0

    while True:
        inp = input("Press Enter when ready to start recording...")
        if inp.lower() == "exit":
            break
        # Record audio
        audio_data = record_audio(duration)
        print("Done recording!")

        # Save audio
        file_name = os.path.join(speaker_dir, f"{session_id}_{speaker_id}_{iteration}.opus")

        save_audio(audio_data, file_name)

        # Transcribe audio
        inputs = prepare_inputs(processor, {"speech": audio_data}, device)

        # Run inference
        transcriptions = run_inference(model, inputs, processor)
        if len(transcriptions) == 0:
            continue
        transcription = transcriptions[0]
        print(f"Transcription: {transcription}")
        recordings.append((iteration, transcription, file_name))
        iteration += 1

    with open(os.path.join(rec_dir, "transcripts.txt"), "w") as f:
        for (iteration, transcription, file_name) in recordings:
            f.write(f"{session_id}_{speaker_id}_{iteration}\t{transcription}\n")
    print("\n=== Recording Session Complete ===")
    print(f"Recorded {iteration} samples")
    print(f"Dataset saved to: {rec_dir}")


def main():
    record_wakeword_samples()


if __name__ == "__main__":
    main()