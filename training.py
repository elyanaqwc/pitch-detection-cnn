import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

class PitchDataset(Dataset):
    def __init__(self, cqt_patches, label_patches):
        self.cqt_patches = cqt_patches
        self.label_patches = label_patches

    def __len__(self):
        return len(self.cqt_patches)

    def __getitem__(self, idx):
        # Get the data
        cqt = self.cqt_patches[idx].astype(np.float32)
        labels = self.label_patches[idx].astype(np.float32)

        # Convert to tensors and add channel dimension for both
        cqt = torch.from_numpy(cqt).unsqueeze(0)      # (88, 128) -> (1, 88, 128)
        labels = torch.from_numpy(labels).unsqueeze(0) # (88, 128) -> (1, 88, 128)

        return cqt, labels

def load_data(result_dir='/content/drive/MyDrive/ECE539 project/Result'):
    """Load and concatenate all processed data files."""
    npz_files = sorted(glob.glob(os.path.join(result_dir, '*.npz')))
    
    all_cqt_patches = []
    all_label_patches = []
    
    print(f"Loading data from {len(npz_files)} files...")
    
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        cqt_patches = data['cqt_patches']
        label_patches = data['label_patches']
        
        all_cqt_patches.append(cqt_patches)
        all_label_patches.append(label_patches)
    
    # Concatenate along the first dimension
    all_cqt_patches = np.concatenate(all_cqt_patches, axis=0)
    all_label_patches = np.concatenate(all_label_patches, axis=0)
    
    print(f"Total patches loaded: {len(all_cqt_patches)}")
    return all_cqt_patches, all_label_patches

def create_data_loaders(cqt_patches, label_patches, batch_size=4, test_size=0.2, val_size=0.5):
    """Split data and create train/val/test loaders."""
    # First split: 80% train, 20% temp
    train_cqt, temp_cqt, train_labels, temp_labels = train_test_split(
        cqt_patches, label_patches, test_size=test_size, random_state=42
    )
    
    # Second split: split temp into val and test
    val_cqt, test_cqt, val_labels, test_labels = train_test_split(
        temp_cqt, temp_labels, test_size=val_size, random_state=42
    )
    
    print(f"Train: {len(train_cqt)} patches")
    print(f"Val: {len(val_cqt)} patches") 
    print(f"Test: {len(test_cqt)} patches")
    
    # Create datasets
    train_dataset = PitchDataset(train_cqt, train_labels)
    val_dataset = PitchDataset(val_cqt, val_labels)
    test_dataset = PitchDataset(test_cqt, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def create_model():
    """Create U-Net model for pitch detection."""
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        activation=None
    )
    return model

def setup_training(model, lr=0.001):
    """Setup model, loss, and optimizer."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    return model, criterion, optimizer, device

def calculate_metrics(predictions, labels, threshold=0.5):
    """Calculate F1, precision, and recall."""
    if torch.is_tensor(predictions):
        predictions = torch.sigmoid(predictions).cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    pred_binary = (predictions > threshold).astype(int)
    pred_flat = pred_binary.flatten()
    labels_flat = labels.flatten()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_flat, pred_flat, average='macro', zero_division=0
    )
    
    return {'f1': f1, 'precision': precision, 'recall': recall}

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, (cqt, labels) in enumerate(train_loader):
        cqt = cqt.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(cqt)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validate model and calculate metrics."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for cqt, labels in val_loader:
            cqt = cqt.to(device)
            labels = labels.to(device)
            
            outputs = model(cqt)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            all_predictions.append(outputs.cpu())
            all_labels.append(labels.cpu())
    
    # Calculate metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = calculate_metrics(all_predictions, all_labels)
    
    return total_loss / len(val_loader), metrics

def save_checkpoint(model, optimizer, epoch, metrics, best_f1, save_dir='checkpoints'):
    """Save model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'best_f1': best_f1
    }
    
    checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch:03d}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Saved checkpoint: {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint and return epoch and best_f1."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_f1 = checkpoint.get('best_f1', 0.0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"✅ Resumed from epoch {epoch}, best F1: {best_f1:.4f}")
    return epoch, best_f1, metrics

def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                num_epochs=5, save_dir='checkpoints', resume_from=None):
    """Complete training loop with resume capability."""
    print("Starting training...")
    best_f1 = 0.0
    start_epoch = 0
    os.makedirs(save_dir, exist_ok=True)
    
    # Resume from checkpoint if specified
    if resume_from:
        start_epoch, best_f1, _ = load_checkpoint(model, optimizer, resume_from)
    
    for epoch in range(start_epoch, num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # Print results
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val F1: {val_metrics["f1"]:.4f}')
        print(f'Val Precision: {val_metrics["precision"]:.4f}')
        print(f'Val Recall: {val_metrics["recall"]:.4f}')
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch+1, val_metrics, best_f1, save_dir)
        
        # Save best model
        current_f1 = val_metrics['f1']
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"🏆 New best F1 score: {best_f1:.4f}")
    
    print(f"\nTraining complete! Best F1 score: {best_f1:.4f}")
    return best_f1

# Main execution
if __name__ == "__main__":
    # Load data
    cqt_patches, label_patches = load_data()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(cqt_patches, label_patches)
    
    # Create and setup model
    model = create_model()
    model, criterion, optimizer, device = setup_training(model)
    
    # Train model (fresh start)
    best_f1 = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50)
    
    # To resume from a specific checkpoint:
    # best_f1 = train_model(model, train_loader, val_loader, criterion, optimizer, device, 
    #                      num_epochs=10, resume_from='checkpoints/model_epoch_003.pth')