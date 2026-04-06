import numpy as np

def calculate_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def calculate_metrics(y_true, y_pred):
    classes = np.unique(y_true)
    metrics_per_class = {}
    
    total_samples = len(y_true)
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0
    
    for c in classes:
        TP = np.sum((y_true == c) & (y_pred == c))
        FP = np.sum((y_true != c) & (y_pred == c))
        FN = np.sum((y_true == c) & (y_pred != c))
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_per_class[c] = {'Precision': precision, 'Recall': recall, 'F1': f1}

        weight = np.sum(y_true == c) / total_samples
        weighted_precision += precision * weight
        weighted_recall += recall * weight
        weighted_f1 += f1 * weight
        
    return {
        'Accuracy': calculate_accuracy(y_true, y_pred),
        'Weighted_Precision': weighted_precision,
        'Weighted_Recall': weighted_recall,
        'Weighted_F1': weighted_f1,
        'Per_Class': metrics_per_class
    }
    
def print_confusion_matrix(y_true, y_pred):
    """
    Generates a cleanly formatted text matrix you can copy-paste into your report.
    """
    classes = sorted(np.unique(y_true))
    matrix = {c: {c_pred: 0 for c_pred in classes} for c in classes}
    
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
        
    print("\n--- Confusion Matrix ---")
    print(f"{'':>10}", end="")
    for c in classes:
        print(f"{c:>10}", end="")
    print()
    
    for c_true in classes:
        print(f"{c_true:>10}", end="")
        for c_pred in classes:
            print(f"{matrix[c_true][c_pred]:>10}", end="")
        print()
    print("-" * 24)