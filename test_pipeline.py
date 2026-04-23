import torch
from model import EmotionCNN

def test_pytorch_pipeline():
    print("[*] Generating dummy data for testing...")
    X = torch.randn(10, 1, 48, 48)
    y = torch.randint(0, 7, (10,))
    
    print("[*] Building model...")
    model = EmotionCNN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("[*] Running 1 epoch of training...")
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    print(f"[✓] Loss after 1 step: {loss.item():.4f}")
    print("[✓] PyTorch Pipeline verified successfully!")

if __name__ == "__main__":
    test_pytorch_pipeline()
