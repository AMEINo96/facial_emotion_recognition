import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# User provided path
BASE_DATA_PATH = r"C:\Users\muham\Downloads\archive"

def get_dataloaders(base_path=BASE_DATA_PATH, batch_size=64):
    """
    Loads FER-2013 from image folders using ImageFolder.
    Structure should be: base_path/train/emotion_name/*.jpg
    """
    train_dir = os.path.join(base_path, 'train')
    test_dir = os.path.join(base_path, 'test')
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found at {test_dir}")

    # Standard FER-2013 images are 48x48 grayscale.
    # We ensure they are 48x48 and single channel (grayscale).
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_set = datasets.ImageFolder(root=train_dir, transform=train_transform)
    # We'll use the 'test' folder for both validation and final testing
    val_set = datasets.ImageFolder(root=test_dir, transform=val_test_transform)
    test_set = datasets.ImageFolder(root=test_dir, transform=val_test_transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    # FER-2013 Labels from ImageFolder classes
    classes = train_set.classes
    print(f"[+] Detected Classes: {classes}")
    print(f"[+] Loaded {len(train_set)} training and {len(val_set)} test samples from {base_path}")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    try:
        tl, vl, tsl = get_dataloaders()
    except Exception as e:
        print(f"[!] Error: {e}")
