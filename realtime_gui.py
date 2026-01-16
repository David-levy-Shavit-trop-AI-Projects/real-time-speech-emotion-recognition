# ======================================================
# Imports & Config
# ======================================================
import numpy as np
import torch
import torch.nn as nn
import librosa
import sounddevice as sd
import tkinter as tk
from tkinter import ttk
import time
from collections import deque

import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# Reliability Parameters (IMPORTANT)
# ======================================================
SILENCE_RMS_THRESHOLD = 0.01
CONFIDENCE_THRESHOLD = 0.45
PREDICTION_INTERVAL = 0.2  # seconds
SMOOTHING_WINDOW = 3

# ======================================================
# Model Definition
# ======================================================
class CNNGRU(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.gru = nn.GRU(64 * 16, 128, batch_first=True)
        self.fc = nn.Linear(128, config.NUM_EMOTIONS)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))

# ======================================================
# Load Model
# ======================================================
model = CNNGRU().to(DEVICE)
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=DEVICE))
model.eval()

# ======================================================
# Feature Extraction
# ======================================================
def extract_log_mel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS
    )
    log_mel = librosa.power_to_db(mel)

    if log_mel.shape[1] < config.MAX_FRAMES:
        log_mel = np.pad(
            log_mel,
            ((0, 0), (0, config.MAX_FRAMES - log_mel.shape[1]))
        )
    else:
        log_mel = log_mel[:, :config.MAX_FRAMES]

    return log_mel

# ======================================================
# Helper Functions
# ======================================================
def is_speech(audio):
    rms = np.sqrt(np.mean(audio ** 2))
    return rms > SILENCE_RMS_THRESHOLD

# ======================================================
# Shared State
# ======================================================
audio_buffer = np.zeros(0, dtype=np.float32)
current_probs = np.zeros(config.NUM_EMOTIONS)
current_emotion = "Waiting..."

emotion_history = deque(maxlen=SMOOTHING_WINDOW)

running = False
stream = None
last_prediction_time = 0

WINDOW_DURATION = 0.6  # seconds
WINDOW_SIZE = int(config.SAMPLE_RATE * WINDOW_DURATION)
CHUNK_SIZE = int(0.05 * config.SAMPLE_RATE)  # 50ms chunks

# ======================================================
# Audio Callback
# ======================================================
def audio_callback(indata, frames, time_info, status):
    global audio_buffer, current_probs, current_emotion
    global last_prediction_time

    if not running:
        return

    audio_buffer = np.concatenate((audio_buffer, indata[:, 0]))
    audio_buffer = audio_buffer[-WINDOW_SIZE:]

    now = time.time()

    if len(audio_buffer) < WINDOW_SIZE:
        return

    if now - last_prediction_time < PREDICTION_INTERVAL:
        return

    last_prediction_time = now

    # Silence detection
    if not is_speech(audio_buffer):
        current_emotion = "Silence"
        current_probs = np.zeros(config.NUM_EMOTIONS)
        emotion_history.clear()
        return

    features = extract_log_mel(audio_buffer)
    x = torch.tensor(features).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]

    max_prob = np.max(probs)

    if max_prob < CONFIDENCE_THRESHOLD:
        current_emotion = "Uncertain"
        current_probs = probs
        return

    pred = np.argmax(probs)
    emotion_history.append(pred)

    smoothed = max(set(emotion_history), key=emotion_history.count)

    current_probs = probs
    current_emotion = config.EMOTIONS[smoothed]

# ======================================================
# GUI
# ======================================================
root = tk.Tk()
root.title("Real-Time Emotion Recognition")
root.geometry("400x420")

label = ttk.Label(root, text="Emotion: ---", font=("Helvetica", 16))
label.pack(pady=20)

bars = []
for e in config.EMOTIONS:
    f = ttk.Frame(root)
    f.pack(fill="x", padx=20, pady=2)

    ttk.Label(f, text=e, width=10).pack(side="left")

    bar = ttk.Progressbar(f, maximum=1.0)
    bar.pack(side="left", fill="x", expand=True)

    bars.append(bar)

def update_gui():
    label.config(text=f"Emotion: {current_emotion}")

    for i, b in enumerate(bars):
        b["value"] = current_probs[i]

    root.after(100, update_gui)

def start():
    global running, stream

    if running:
        return

    running = True

    stream = sd.InputStream(
        channels=1,
        samplerate=config.SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    )
    stream.start()

def stop():
    global running, stream

    running = False

    if stream is not None:
        stream.stop()
        stream.close()
        stream = None

    label.config(text="Emotion: Stopped")

ttk.Button(root, text="Start", command=start).pack(pady=5)
ttk.Button(root, text="Stop", command=stop).pack(pady=5)

def on_close():
    stop()
    root.destroy()

update_gui()
root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
