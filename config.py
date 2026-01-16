# ============================
# Audio & Feature Parameters
# ============================
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 400
HOP_LENGTH = 160
MAX_FRAMES = 64

# ============================
# Model Parameters
# ============================
CNN_CHANNELS = [16, 32, 64]
GRU_HIDDEN_SIZE = 128
MODEL_PATH = "cnn_gru_ravdess.pth"

# ============================
# Emotions (RAVDESS)
# ============================
EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]

NUM_EMOTIONS = len(EMOTIONS)

EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {i: e for e, i in EMOTION_TO_IDX.items()}

# ============================
# Streaming (GUI)
# ============================
CHUNK_DURATION = 0.1
WINDOW_DURATION = 1.0
STEP_DURATION = 0.5
