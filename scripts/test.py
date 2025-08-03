import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import os
import glob
import time

from train import create_data_loaders, create_model


def load_best_model(model_path, device):
    """Load the best trained model."""
    print(f"Loading model from {model_path}")
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    return model


def evaluate_model(model, test_loader, device):
    """Evaluate model and return predictions and labels."""
    print("Running evaluation...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (cqt, labels) in enumerate(test_loader):
            cqt = cqt.to(device)
            labels = labels.to(device)
            
            outputs = model(cqt)
            predictions = torch.sigmoid(outputs)
            
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            
            if batch_idx % 10 == 0:
                print(f"  Processed {batch_idx}/{len(test_loader)} batches")
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    print(f"Evaluated {len(all_predictions)} samples")
    return all_predictions, all_labels


def calculate_metrics(predictions, labels, threshold=0.5):
    """Calculate F1, precision, recall, accuracy."""
    pred_binary = (predictions > threshold).numpy().astype(int)
    labels_np = labels.numpy().astype(int)
    
    pred_flat = pred_binary.flatten()
    labels_flat = labels_np.flatten()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_flat, pred_flat, average='binary', zero_division=0
    )
    
    accuracy = (pred_flat == labels_flat).mean()
    tn, fp, fn, tp = confusion_matrix(labels_flat, pred_flat).ravel()
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def find_best_threshold(predictions, labels):
    """Find optimal threshold based on F1 score."""
    print("Finding best threshold...")
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        metrics = calculate_metrics(predictions, labels, threshold)
        print(f"  Threshold {threshold}: F1={metrics['f1']:.4f}")
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = threshold
    
    print(f"Best threshold: {best_threshold} (F1: {best_f1:.4f})")
    return best_threshold


def visualize_predictions(predictions, labels, num_samples=3, threshold=0.5):
    """Show a few prediction examples."""
    print(f"Creating visualization for {num_samples} samples...")
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(num_samples, predictions.shape[0])):
        # Handle tensor dimensions properly
        if len(predictions.shape) == 4:
            pred = predictions[i, 0].numpy()
            label = labels[i, 0].numpy()
        else:
            pred = predictions[i].numpy()
            label = labels[i].numpy()
            
        pred_binary = (pred > threshold).astype(int)
        
        axes[i, 0].imshow(label, aspect='auto', origin='lower', cmap='Blues')
        axes[i, 0].set_title(f'Ground Truth {i+1}')
        
        axes[i, 1].imshow(pred, aspect='auto', origin='lower', cmap='Blues')
        axes[i, 1].set_title(f'Prediction {i+1}')
        
        axes[i, 2].imshow(pred_binary, aspect='auto', origin='lower', cmap='Blues')
        axes[i, 2].set_title(f'Binary (t={threshold})')
        
        if i == num_samples - 1:
            for j in range(3):
                axes[i, j].set_xlabel('Time Frames')
    
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'test_results.png'")
    plt.show()


def main():
    """Main testing function."""
    print("Starting model testing...")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find model
    model_path = 'checkpoints/best_model.pth'
    if not os.path.exists(model_path):
        checkpoint_files = glob.glob('checkpoints/*.pth')
        if checkpoint_files:
            model_path = checkpoint_files[-1]
            print(f"Using: {model_path}")
        else:
            print("No model found!")
            return
    
    # Load data and model
    try:
        _, _, test_loader = create_data_loaders(batch_size=32, subset_fraction=0.1)
        model = load_best_model(model_path, device)
    except Exception as e:
        print(f"Error loading: {e}")
        return
    
    # Test model
    predictions, labels = evaluate_model(model, test_loader, device)
    best_threshold = find_best_threshold(predictions, labels)
    final_metrics = calculate_metrics(predictions, labels, best_threshold)
    
    # Show results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Best Threshold: {best_threshold}")
    print(f"F1 Score: {final_metrics['f1']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print("="*50)
    
    # Save results
    with open('test_results.txt', 'w') as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Best Threshold: {best_threshold}\n")
        f.write(f"F1 Score: {final_metrics['f1']:.4f}\n")
        f.write(f"Precision: {final_metrics['precision']:.4f}\n")
        f.write(f"Recall: {final_metrics['recall']:.4f}\n")
        f.write(f"Accuracy: {final_metrics['accuracy']:.4f}\n")
    
    visualize_predictions(predictions, labels, num_samples=3, threshold=best_threshold)
    print("Results saved to 'test_results.txt' and 'test_results.png'")


if __name__ == "__main__":
    main()