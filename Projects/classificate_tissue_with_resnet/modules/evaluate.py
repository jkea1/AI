import torch

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    avg_val_loss = running_loss / len(val_loader)
    
    return avg_val_loss  # float 하나만 반환