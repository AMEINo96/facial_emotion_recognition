import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from data_loader import get_dataloaders
from model import EmotionCNN

# Configuration
MODEL_PATH = "emotion_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    # 1. Load Data
    try:
        _, _, test_loader = get_dataloaders()
        classes = test_loader.dataset.classes
    except Exception as e:
        print(f"[!] Evaluation failed: {e}")
        return

    # 2. Load Model
    model = EmotionCNN()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"[+] Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"[!] Error: Model file {MODEL_PATH} not found. Train the model first.")
        return
        
    model.to(DEVICE)
    model.eval()

    # 3. Predict on Test Set
    all_preds = []
    all_labels = []

    print("[*] Evaluating on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # 5. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
    print("[+] Confusion matrix saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    evaluate()
