# filename: notebook/final_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# 1. Setup Data Paths
DATA_DIR = '../data' 

# 2. Image Transformations (Critical for Medical Scans)
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Load Dataset
print("Loading dataset from folder...")
full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)

# !!! CRITICAL FOR ACCURACY !!! 
# This forces PyTorch to lock the classes in the exact same order as your backend
full_dataset.classes = ['COVID', 'Normal', 'Pneumonia', 'Tuberculosis']
full_dataset.class_to_idx = {'COVID': 0, 'Normal': 1, 'Pneumonia': 2, 'Tuberculosis': 3}

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

# Using num_workers=0 to prevent multi-processing bugs on Windows
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

# 4. Initialize Model (ResNet18)
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4) # 4 Classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 5. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training Loop
print(f"Starting Training on {device}...")
for epoch in range(5):
    model.train()
    running_loss = 0.0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Print progress every 10 batches so you know it's not frozen
        if i % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
    
    print(f"--- Epoch {epoch+1} Complete! Average Loss: {running_loss/len(train_loader):.4f} ---")

# 7. Create backend/models folder if it doesn't exist
os.makedirs('../backend/models', exist_ok=True)

# 8. Save for Backend
torch.save(model.state_dict(), "../backend/models/medical_cnn.pth")
print("\nProject Complete! Weights saved to backend/models/medical_cnn.pth")