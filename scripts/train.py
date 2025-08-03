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
import psutil
import time
import gc
import pickle
from collections import defaultdict

class PitchDataset(Dataset):   
    def __init__(self, npz_files, file_sample_mapping, indices=None, max_cache_files=10):
        self.npz_files = npz_files
        self.file_sample_mapping = file_sample_mapping
        self.indices = indices if indices is not None else list(range(len(file_sample_mapping)))
        self.max_cache_files = max_cache_files
        
        # Group samples by file to enable batch loading
        self.file_to_samples = defaultdict(list)
        for idx in self.indices:
            file_idx, sample_idx = self.file_sample_mapping[idx]
            self.file_to_samples[file_idx].append((idx, sample_idx))
        
        # Cache for loaded files (LRU-style)
        self.file_cache = {}
        self.file_access_order = []
        
        print(f"Dataset covers {len(self.file_to_samples)} unique files")

    def _load_file_if_needed(self, file_idx):
        """Load file into cache if not already loaded."""
        if file_idx in self.file_cache:
            # Move to end (most recently used)
            self.file_access_order.remove(file_idx)
            self.file_access_order.append(file_idx)
            return
        
        # Load the file
        npz_file = self.npz_files[file_idx]
        data = np.load(npz_file, allow_pickle=True)
        
        # Cache it
        self.file_cache[file_idx] = {
            'cqt_patches': data['cqt_patches'],
            'label_patches': data['label_patches']
        }
        self.file_access_order.append(file_idx)
        
        # Remove oldest files if cache is full
        while len(self.file_cache) > self.max_cache_files:
            oldest_file = self.file_access_order.pop(0)
            del self.file_cache[oldest_file]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the actual sample index
        actual_idx = self.indices[idx]
        file_idx, sample_idx = self.file_sample_mapping[actual_idx]
        
        # Load file if needed (smart caching)
        self._load_file_if_needed(file_idx)
        
        # Get the specific sample from cache
        cached_data = self.file_cache[file_idx]
        cqt = cached_data['cqt_patches'][sample_idx].astype(np.float32)
        labels = cached_data['label_patches'][sample_idx].astype(np.float32)

        # Convert to tensors and add channel dimension
        cqt = torch.from_numpy(cqt).unsqueeze(0)      # (88, 128) -> (1, 88, 128)
        labels = torch.from_numpy(labels).unsqueeze(0) # (88, 128) -> (1, 88, 128)

        return cqt, labels

def load_data(result_dir='data', cache_file='file_index.pkl', max_memory_percent=70):
    npz_files = sorted(glob.glob(os.path.join(result_dir, '*.npz')))
    print(f"Found {len(npz_files)} NPZ files")
    
    # Check current memory usage
    memory_percent = psutil.virtual_memory().percent
    print(f"Current memory usage: {memory_percent:.1f}%")
    
    if memory_percent > max_memory_percent:
        print(f"Memory usage is high ({memory_percent}%). Consider closing other programs.")
        time.sleep(3)
    
    # Check if cache exists and is up to date
    cache_path = os.path.join(result_dir, cache_file)
    if os.path.exists(cache_path):
        print("Loading cached file index (super fast!)...")
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
            
        # Verify cache is still valid (same files)
        if cached_data['npz_files'] == npz_files:
            print("Cache is valid, using cached index")
            print(f"Total samples: {len(cached_data['file_sample_mapping'])}")
            return npz_files, cached_data['file_sample_mapping']
        else:
            print("Cache outdated (files changed), rebuilding...")
    
    # Build index (only happens first time or when files change)
    print("Building file index...")
    file_sample_mapping = []
    
    for i, npz_file in enumerate(npz_files):
        if i % 100 == 0:
            print(f"  Scanned {i}/{len(npz_files)} files...")
        
        with np.load(npz_file, allow_pickle=True) as data:
            num_samples = data['cqt_patches'].shape[0]
            
        for sample_idx in range(num_samples):
            file_sample_mapping.append((i, sample_idx))
    
    # Save cache for next time
    cache_data = {
        'npz_files': npz_files,
        'file_sample_mapping': file_sample_mapping
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"Saved index cache to {cache_path}")
    print(f"Total samples: {len(file_sample_mapping)}")
    
    return npz_files, file_sample_mapping


def create_data_loaders(result_dir='data', batch_size=32, test_size=0.2, 
                               val_size=0.5, subset_fraction=0.1, 
                               use_preloaded=True, max_samples_in_ram=10000):
    """Create data loaders with improved caching."""
    
    print("Creating data loaders...")
    
    # Get files and cached mapping
    npz_files, file_sample_mapping = load_data(result_dir)
    
    # Use subset of data
    total_samples = int(len(file_sample_mapping) * subset_fraction)
    all_indices = list(range(total_samples))
    
    print(f"Using {subset_fraction*100}% of data: {total_samples:,} samples")
    
    # Create train/val/test splits
    train_indices, temp_indices = train_test_split(all_indices, test_size=test_size, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=val_size, random_state=42)
    
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    train_dataset = PitchDataset(npz_files, file_sample_mapping, train_indices, max_cache_files=20)
    val_dataset = PitchDataset(npz_files, file_sample_mapping, val_indices, max_cache_files=10)
    test_dataset = PitchDataset(npz_files, file_sample_mapping, test_indices, max_cache_files=5)
    
    # Create data loaders with multiple workers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
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


def monitor_memory_usage():
    """Print current memory usage."""
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent:.1f}% ({memory.used/(1024**3):.1f}GB/{memory.total/(1024**3):.1f}GB)")


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
            # Monitor memory usage periodically
            if batch_idx % 100 == 0:
                monitor_memory_usage()
    
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
    print(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint and return epoch and best_f1."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_f1 = checkpoint.get('best_f1', 0.0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"Resumed from epoch {epoch}, best F1: {best_f1:.4f}")
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
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val F1: {val_metrics["f1"]:.4f}')
        print(f'Val Precision: {val_metrics["precision"]:.4f}')
        print(f'Val Recall: {val_metrics["recall"]:.4f}')
        
        save_checkpoint(model, optimizer, epoch+1, val_metrics, best_f1, save_dir)
        
        current_f1 = val_metrics['f1']
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"New best F1 score: {best_f1:.4f}")
    
    print(f"\nTraining complete! Best F1 score: {best_f1:.4f}")
    return best_f1


if __name__ == "__main__":
    print("Starting training...")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=64,           
        subset_fraction=0.1,    
        use_preloaded=True,      
        max_samples_in_ram=15000 
    )
    
    model = create_model()
    model, criterion, optimizer, device = setup_training(model)
    
    best_f1 = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=15, resume_from='checkpoints/model_epoch_010.pth')
    
  