import torch

def calculate_metrics(y_pred, y_true, threshold=0.5):
    """
    Calculate IoU, Dice, FPR, and FNR metrics.
    
    Args:
        y_pred: Predicted output (logits or probabilities).
        y_true: Ground truth labels.
        threshold: Threshold to binarize predictions (default 0.5).
        
    Returns:
        A dictionary with IoU, Dice, FPR, and FNR.
    """
    y_pred = (y_pred > threshold).float()  # Binarize predictions
    y_true = y_true.float()
    
    # Compute TP, FP, FN
    TP = torch.sum((y_pred == 1) & (y_true == 1)).item()
    FP = torch.sum((y_pred == 1) & (y_true == 0)).item()
    FN = torch.sum((y_pred == 0) & (y_true == 1)).item()
    TN = torch.sum((y_pred == 0) & (y_true == 0)).item()

    # Compute metrics
    iou = TP / (TP + FP + FN + 1e-7)
    dice = 2 * TP / (2 * TP + FP + FN + 1e-7)
    fpr = FP / (TP + FN + 1e-7)
    fnr = FN / (TP + FN + 1e-7)
    
    return {
        "IoU": iou,
        "Dice": dice,
        "FPR": fpr,
        "FNR": fnr,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN
    }


# Set the model to evaluation mode
model.eval()

# Initialize metrics
all_metrics = []

# Iterate over the test dataloader
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        predictions = torch.sigmoid(outputs)  # If using sigmoid for binary segmentation
        
        # Compute metrics
        batch_metrics = calculate_metrics(predictions, labels)
        all_metrics.append(batch_metrics)

# Compute average metrics over the dataset
average_metrics = {key: sum(metric[key] for metric in all_metrics) / len(all_metrics) for key in all_metrics[0]}

# Display the results
print("Evaluation Metrics:")
for metric, value in average_metrics.items():
    print(f"{metric}: {value:.4f}")
