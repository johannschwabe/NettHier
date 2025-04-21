import os
import torch
import librosa
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import Dataset

from config import MODEL_ID


def check_device():
    """Check if GPU is available and return the appropriate device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU instead")
    return device


def load_mls_dataset(base_path, num_samples=None):
    """
    Create a dataset from MLS data.

    Args:
        base_path: Path to MLS dataset
        num_samples: Number of samples to use (None for all)

    Returns:
        Hugging Face Dataset or None if failed
    """
    # Load metadata
    metadata_path = os.path.join(base_path, "transcripts.txt")

    if not os.path.exists(metadata_path):
        print(f"Metadata file not found at {metadata_path}")
        return None

    # Read the metadata file
    metadata = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                file_id = parts[0]
                transcript = parts[1]

                # MLS file paths for opus files
                split_id = file_id.split('_')
                speaker_id = split_id[0]
                book_id = split_id[1]
                audio_path = os.path.join(base_path, "audio", speaker_id, book_id, f"{file_id}.opus")

                if os.path.exists(audio_path):
                    metadata.append({"file_id": file_id, "path": audio_path, "sentence": transcript})
                else:
                    print(f"Audio file not found: {audio_path}")

    if not metadata:
        print("No valid files found in the metadata")
        return None

    print(f"Found {len(metadata)} valid files")

    # Create a DataFrame and then convert to a Hugging Face Dataset
    df = pd.DataFrame(metadata)
    dataset = Dataset.from_pandas(df)

    # Select a subset of samples
    if num_samples and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    return dataset


def speech_file_to_array_fn(batch):
    """
    Preprocess audio files in the dataset.

    Args:
        batch: Dataset batch containing path and sentence

    Returns:
        Processed batch with speech array
    """
    try:
        speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
        batch["speech"] = speech_array
        batch["sentence"] = batch["sentence"].upper()  # Uppercase for matching model expectations
    except Exception as e:
        print(f"Error loading audio file {batch['path']}: {e}")
        # Use a placeholder to avoid breaking the dataset
        batch["speech"] = torch.zeros(16000).numpy()
    return batch


def load_model_and_processor(device):
    """
    Load the speech recognition model and processor.

    Args:
        device: Torch device to use

    Returns:
        Tuple of (processor, model)
    """
    print("Loading model and processor...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    return processor, model


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
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True
    )

    # Then move to GPU with correct dtype
    inputs = {
        'input_values': inputs.input_values.to(device, dtype=torch.float16),
        'attention_mask': inputs.attention_mask.to(device) if inputs.attention_mask is not None else None
    }

    return inputs


def run_inference(model, inputs, processor):
    """
    Run inference on audio inputs.

    Args:
        model: Wav2Vec2ForCTC model
        inputs: Prepared inputs dictionary
        processor: Wav2Vec2Processor

    Returns:
        List of predicted sentences
    """
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)

        # Get plain text transcriptions
        predicted_sentences = processor.batch_decode(predicted_ids)

    return predicted_sentences


def display_results(dataset, predicted_sentences):
    """
    Display the transcription results.

    Args:
        dataset: Dataset with reference sentences
        predicted_sentences: List of model predictions
    """
    for i, predicted_sentence in enumerate(predicted_sentences):
        print("-" * 100)
        print(f"Sample {i + 1}: {dataset[i]['file_id']}")
        print("Reference:", dataset[i]["sentence"])
        print("Prediction:", predicted_sentence)


def main():
    """Main function to run the speech recognition pipeline."""
    # Set model configuration
    LANG_ID = "de"
    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-german"
    SAMPLES = 10

    # MLS German dataset path
    mls_german_path = "/home/js/Downloads/mls_german_opus/train"

    # Check device
    device = check_device()

    # Load and preprocess the MLS dataset
    print("Loading MLS German dataset...")
    mls_dataset = load_mls_dataset(mls_german_path, SAMPLES)

    if mls_dataset is None:
        print("Failed to load MLS German dataset")
        return

    # Preprocess the dataset
    mls_dataset = mls_dataset.map(speech_file_to_array_fn)

    # Load model and processor
    processor, model = load_model_and_processor(device)

    # Prepare inputs
    inputs = prepare_inputs(processor, mls_dataset, device)

    # Run inference
    predicted_sentences = run_inference(model, inputs, processor)

    # Display results
    display_results(mls_dataset, predicted_sentences)


if __name__ == "__main__":
    main()