"""
Configuration settings for wake word detection system.
"""

# Audio processing settings
SAMPLE_RATE = 16000  # 16kHz
WINDOW_SIZE_MS = 500  # 500ms window
HOP_LENGTH_MS = 100  # 100ms hop
WINDOW_SIZE_SAMPLES = WINDOW_SIZE_MS * SAMPLE_RATE // 1000


# Feature extraction settings
INPUT_FEATURES = 13  # Number of MFCC features
FFT_SIZE = 512
PREEMPHASIS_COEFF = 0.97

# Model parameters
HIDDEN_SIZE = 24
NUM_LAYERS = 3

# Detection settings
DETECTION_THRESHOLD = 0.5  # Probability threshold for positive detection
ACTIVATION_FRAMES = 3  # Number of consecutive positive frames needed for detection

# Model paths
ESP32_MODEL_DIR = "esp32_model"  # Directory containing exported models
MODEL_PATH = f"{ESP32_MODEL_DIR}/wakeword_model.pt"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-german"
MLS_PATH = "/home/js/Downloads/mls_german_opus/train"
LOCAL_RECORDINGS = "/home/js/PycharmProjects/PythonProject/recorded"


# Wake word
WAKEWORD = " danke"

NO_CACHE = True
SAMPLES = None