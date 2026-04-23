import cv2
import torch
import numpy as np
from model import EmotionCNN
from torchvision import transforms
from PIL import Image

# Configuration
MODEL_PATH = "emotion_model.pth"
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path):
    model = EmotionCNN()
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"[+] Model loaded from {path}")
    except Exception as e:
        print(f"[!] Warning: Could not load model from {path}. Check if it exists.")
        print(f"    Error: {e}")
        print("    Running with randomly initialized weights for demonstration.")
    model.to(DEVICE)
    model.eval()
    return model

def real_time_detection():
    # 1. Initialize Face Detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 2. Load Model
    model = load_model(MODEL_PATH)
    
    # 3. Define Preprocessing
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 4. Open Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Error: Could not open webcam.")
        return

    print("[*] Starting camera. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract and preprocess face
            face_roi = gray[y:y+h, x:x+w]
            face_pil = Image.fromarray(face_roi)
            face_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)
            
            # Predict
            with torch.no_grad():
                output = model(face_tensor)
                probs = torch.softmax(output, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_idx].item()
                
            emotion = EMOTIONS[pred_idx]
            label = f"{emotion} ({confidence*100:.1f}%)"
            
            # Display emotion label
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            
        cv2.imshow('Facial Emotion Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_detection()
