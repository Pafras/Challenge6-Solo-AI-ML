import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader


from show_batch import CLASSES, FER2013

class TinyCNN(nn.Module):
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16 * 12 * 12, n_classes),)  
    def forward(self, x):
        return self.net(x)

def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_correct = total_n = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n += len(labels)
    return total_loss / total_n, total_correct / total_n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = total_n = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n += len(labels)
    return total_loss / total_n, total_correct / total_n


if __name__ == "__main__":
    torch.manual_seed(42)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)

    train_loader = DataLoader(FER2013("train"), batch_size=64, shuffle=True)
    valid_loader = DataLoader(FER2013("valid"), batch_size=64, shuffle=False)

    model = TinyCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 6):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion=criterion, optimizer=optimizer, device=device)
        va_loss, va_acc = evaluate(model, valid_loader, criterion=criterion, device=device)
        print(f"Epoch {epoch}:  Train: {tr_loss:.4f}, Train Acc: {tr_acc:.3f}" f"Valid: {va_loss:.4f} / {va_acc:.3f}")
