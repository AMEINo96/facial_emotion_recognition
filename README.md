# Facial Emotion Recognition (RTX 3050 Ti Accelerated)

A real-time facial emotion recognition system using a custom CNN architecture in PyTorch, accelerated by NVIDIA GPU (CUDA).

## Features
- **Real-time Detection**: Uses OpenCV and a trained CNN to detect emotions via webcam.
- **High Performance**: Optimized for NVIDIA RTX 3050 Ti using CUDA 12.1.
- **Emotion Classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
- **Detailed Reporting**: Includes scripts to generate classification reports and Word documents.

## Performance
- **Accuracy**: 63% on FER-2013 dataset (Human-level accuracy is ~65%).
- **GPU Accelerated**: Full CUDA support enabled.

## Setup
1. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install opencv-python pandas matplotlib tqdm python-docx scikit-learn seaborn
   ```
2. Train the model:
   ```bash
   python train.py
   ```
3. Run real-time detection:
   ```bash
   python real_time_detection.py
   ```

## Files
- `model.py`: CNN Architecture definition.
- `data_loader.py`: Data pipeline using `ImageFolder`.
- `train.py`: Training loop with learning rate scheduler.
- `real_time_detection.py`: OpenCV webcam inference script.
- `evaluate.py`: Metrics and confusion matrix generator.
- `generate_report.py`: Word document report generator.
