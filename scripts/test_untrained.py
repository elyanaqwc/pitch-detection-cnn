import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from train import create_data_loaders, create_model

def test_random_model():
    """Test untrained model to get baseline metrics."""
    print("Testing untrained (random) U-Net model...")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create untrained model (random weights)
    model = create_model()
    model = model.to(device)
    model.eval()
    print("Created untrained model with random weights")
    
    # Load test data
    try:
        _, _, test_loader = create_data_loaders(batch_size=32, subset_fraction=0.05)  # Small subset for speed
        print(f"Loaded test data: {len(test_loader)} batches")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Run evaluation
    print("Running evaluation on random model...")
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (cqt, labels) in enumerate(test_loader):
            cqt = cqt.to(device)
            labels = labels.to(device)
            
            # Get random predictions
            outputs = model(cqt)
            predictions = torch.sigmoid(outputs)
            
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            
            if batch_idx % 10 == 0:
                print(f"  Processed {batch_idx}/{len(test_loader)} batches")
            
            if batch_idx >= 20: 
                break
    
    # Concatenate results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    print(f"Evaluated {len(all_predictions)} samples")
    
    # Test different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print("\nRandom model performance:")
    print("="*50)
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        # Calculate metrics
        pred_binary = (all_predictions > threshold).numpy().astype(int)
        labels_np = all_labels.numpy().astype(int)
        
        pred_flat = pred_binary.flatten()
        labels_flat = labels_np.flatten()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_flat, pred_flat, average='binary', zero_division=0
        )
        
        accuracy = (pred_flat == labels_flat).mean()
        
        print(f"Threshold {threshold}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Acc={accuracy:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print("="*50)
    print(f"Best random performance: F1={best_f1:.4f} at threshold {best_threshold}")
    print("="*50)
    
    # Show class distribution
    labels_flat = all_labels.numpy().flatten()
    positive_ratio = labels_flat.mean()
    print(f"Dataset info:")
    print(f"  Positive class ratio: {positive_ratio:.3f} ({positive_ratio*100:.1f}%)")
    print(f"  Negative class ratio: {1-positive_ratio:.3f} ({(1-positive_ratio)*100:.1f}%)")
    
    print(f"\nComparison:")
    print(f"  Untrained model F1:  {best_f1:.4f}")
    print(f"  Trained F1:   0.8262")
    print(f"  Improvement:       {0.8262/best_f1:.1f}x better")

if __name__ == "__main__":
    test_random_model()