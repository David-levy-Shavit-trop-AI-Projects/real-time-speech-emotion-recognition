# Real-Time Speech Emotion Recognition (SER)

**Authors:** [Shavit Trop](https://linkedin.com/in/shavit-trop) & [David Levy](https://www.linkedin.com/in/dudi-levy)  

This project implements a **real-time Speech Emotion Recognition (SER)** with GUI system as part of an **BSc in Computer Science – Deep Learning course final project**.

The focus of the project is **engineering a low-latency, real-time pipeline** rather than pushing state-of-the-art research accuracy. The system is built using **PyTorch** and performs **live emotion classification from microphone input**.

---

## 🎯 Project Goals

- Train a deep learning model to recognize emotions from speech
- Perform **real-time inference** from a microphone stream
- Emphasize **engineering quality**, modularity, and reproducibility
- Separate **training** and **deployment (GUI)** workflows

---

## 📊 Dataset: [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

**Key properties:**
- 1440 files
- 24 Professional actors (12 male & 12 female)
- Clean studio-quality recordings
- Speech-only subset used
- 8 emotion classes:
  - Neutral
  - Calm
  - Happy
  - Sad
  - Angry
  - Fearful
  - Disgust
  - Surprised

The dataset is downloaded automatically using the Kaggle API:

```python
import kagglehub
path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
```

---

## 🧠 Model Architecture

The model is designed for **real-time performance**:

- **Log-Mel Spectrogram** input
- **CNN feature extractor** (spatial patterns)
- **GRU temporal model** (short-term dynamics)
- Lightweight architecture for low-latency inference

All audio, feature, and model hyperparameters are centralized in `config.py` to ensure consistency between training and inference.

---

## 🗂 Project Structure

```
project/
│
├── cnn_gru_ravdess.pth   # Trained model (ignored by git)
├── config.py             # Shared configuration
├── realtime_gui.py       # Live microphone GUI
├── requirements.txt      # dependancies list
├── training_model.ipynb  # Training, evaluation, plots
└── README.md
```

---

## 🚀 How to Run the Project

### prerequisites

- Python 3

---

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```
---

### 2️⃣ Train the Model (First time users)

Open and "Run All":

```
training_model.ipynb
```

This notebook will:
- Downloads the RAVDESS dataset
- Extracts log-mel features
- Trains the CNN-GRU model
- Plots:
  - Training & validation loss/accuracy
  - Confusion matrix
  - Classification report
- Saves the trained model to:

```
cnn_gru_ravdess.pth
```

---

### 3️⃣ Run the Real-Time GUI

Only after the training is complete (at least once), run the following command on terminal:

```
python realtime_gui.py
```

This command will:
- Loads the trained model
- Captures live audio from the microphone
- Performs streaming inference
- Displays real-time emotion probabilities

> You **do not need to retrain** the model to run the GUI.

---

## 🧪 Engineering Highlights

- Centralized configuration (`config.py`)
- Train–inference parameter consistency
- Modular notebooks for faster demos
- Real-time sliding window inference
- Clean separation of concerns

---

## 🎓 Academic Context

This project was developed as a **final project** in a graduate-level **Deep Learning** course, with emphasis on:

- Practical deep learning systems
- Reproducible experimentation
- Real-time AI deployment considerations

---

## 📌 Notes

- Trained model weights are excluded from version control (`.gitignore`)
- For reproducibility, retrain the model locally
- Accuracy is constrained by real-time requirements and dataset size

---

## 📬 Future Improvements

- Voice Activity Detection (VAD)
- Model quantization / TorchScript export
- Latency and FPS monitoring
- Noise-robust feature extraction
