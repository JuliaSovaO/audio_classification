import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11

class CleanVisualizer:
    def __init__(self, results_dir="visualization_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
    
    def plot_1_singular_value_spectrum(self, singular_values):
        """Plot 1: Singular value spectrum"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        components = np.arange(1, len(singular_values) + 1)
        ax.semilogy(components, singular_values, 'b-o', linewidth=2, markersize=4)
        ax.set_xlabel('Component Index', fontsize=12)
        ax.set_ylabel('Singular Value (log scale)', fontsize=12)
        ax.set_title('Singular Value Spectrum', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=25, color='r', linestyle='--', linewidth=2, label='k=25')
        
        # Add annotations
        ax.annotate(f'σ₁ = {singular_values[0]:.2f}',
                   xy=(1, singular_values[0]), xytext=(10, singular_values[0]*0.7),
                   arrowprops=dict(arrowstyle='->', color='green'))
        ax.annotate(f'σ₂₅ = {singular_values[24]:.4f}',
                   xy=(25, singular_values[24]), xytext=(35, singular_values[24]*3),
                   arrowprops=dict(arrowstyle='->', color='green'))
        
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.results_dir / '1_singular_value_spectrum.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Saved: 1_singular_value_spectrum.png")
    
    def plot_2_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot 2: Confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_xlabel('Predicted', fontweight='bold')
        ax1.set_ylabel('True', fontweight='bold')
        ax1.set_title('Confusion Matrix (Counts)', fontweight='bold')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Normalized
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='RdYlBu_r',
                   xticklabels=class_names, yticklabels=class_names, ax=ax2, vmin=0, vmax=1)
        ax2.set_xlabel('Predicted', fontweight='bold')
        ax2.set_ylabel('True', fontweight='bold')
        ax2.set_title('Confusion Matrix (Normalized)', fontweight='bold')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / '2_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Saved: 2_confusion_matrix.png")
        return cm
    
    def plot_3_misclassification_analysis(self, y_true, y_pred, class_names):
        """Plot 3: Misclassification analysis"""
        cm = confusion_matrix(y_true, y_pred)
        misclass_rate = 1 - np.diag(cm) / cm.sum(axis=1)
        
        # Sort by misclassification rate
        sorted_idx = np.argsort(misclass_rate)[::-1]
        sorted_classes = [class_names[i] for i in sorted_idx]
        sorted_rates = misclass_rate[sorted_idx]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['red' if r > 0.7 else 'orange' if r > 0.5 else 'green' for r in sorted_rates]
        bars = ax.barh(range(len(sorted_classes)), sorted_rates, color=colors, alpha=0.7)
        ax.set_yticks(range(len(sorted_classes)))
        ax.set_yticklabels(sorted_classes)
        ax.set_xlabel('Misclassification Rate', fontweight='bold')
        ax.set_title('Misclassification Rate by Class', fontweight='bold')
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color='red', linestyle='--', label='50% threshold')
        
        for i, (bar, rate) in enumerate(zip(bars, sorted_rates)):
            ax.text(rate + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{rate*100:.1f}%', va='center')
        
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.results_dir / '3_misclassification_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Saved: 3_misclassification_analysis.png")
    
    def plot_4_class_performance(self, y_true, y_pred, class_names):
        """Plot 4: Per-class precision, recall, F1"""
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=class_names
        )
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Grouped bar chart
        x = np.arange(len(class_names))
        width = 0.25
        
        ax1.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax1.bar(x, recall, width, label='Recall', alpha=0.8)
        ax1.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        ax1.set_xlabel('Class', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('Per-Class Performance Metrics', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_names, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Class distribution
        colors = ['#2ecc71' if s > np.mean(support) else '#e74c3c' for s in support]
        ax2.barh(range(len(class_names)), support, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(class_names)))
        ax2.set_yticklabels(class_names)
        ax2.set_xlabel('Number of Samples', fontweight='bold')
        ax2.set_title('Class Distribution (Test Set)', fontweight='bold')
        ax2.axvline(x=np.mean(support), color='blue', linestyle='--', label=f'Mean: {np.mean(support):.0f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / '4_class_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Saved: 4_class_performance.png")
        
        # Print report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        for name, p, r, f, s in zip(class_names, precision, recall, f1, support):
            print(f"{name:12} | P:{p:.3f} R:{r:.3f} F1:{f:.3f} | samples:{s}")
        print("="*60)
    
    def plot_5_dimension_comparison(self, X_original, X_reduced):
        """Plot 5: Distance distribution comparison"""
        np.random.seed(42)
        n = min(300, len(X_original))
        idx = np.random.choice(len(X_original), n, replace=False)
        
        X_orig_sample = X_original[idx]
        X_red_sample = X_reduced[idx]
        
        # Compute pairwise distances
        print("   Computing distances...")
        dist_orig, dist_red = [], []
        for i in range(min(80, n)):
            for j in range(i+1, min(150, n)):
                dist_orig.append(np.linalg.norm(X_orig_sample[i] - X_orig_sample[j]))
                dist_red.append(np.linalg.norm(X_red_sample[i] - X_red_sample[j]))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.hist(dist_orig, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Euclidean Distance', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title(f'Original Space (4000 dims)\nMean = {np.mean(dist_orig):.2f}', fontweight='bold')
        ax1.axvline(np.mean(dist_orig), color='red', linestyle='--')
        
        ax2.hist(dist_red, bins=40, alpha=0.7, color='coral', edgecolor='black')
        ax2.set_xlabel('Euclidean Distance', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title(f'Reduced Space (25 dims)\nMean = {np.mean(dist_red):.2f}', fontweight='bold')
        ax2.axvline(np.mean(dist_red), color='red', linestyle='--')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / '5_dimension_reduction_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Saved: 5_dimension_reduction_comparison.png")


def main():
    print("\n" + "="*60)
    print("GENERATING 5 REQUESTED PLOTS")
    print("="*60 + "\n")
    
    # Load data
    print("Loading data...")
    try:
        X_train = np.load('X_train.npy')
        y_train = np.load('y_train.npy', allow_pickle=True)
        y_test = np.load('y_test.npy', allow_pickle=True)
        print("✓ Loaded X_train, y_train, y_test")
    except Exception as e:
        print(f"Error loading: {e}")
        return
    
    # Load reduced data and predictions
    try:
        X_train_reduced = np.load('X_train_reduced.npy')
        X_test_reduced = np.load('X_test_reduced.npy')
        y_pred = np.load('y_pred.npy', allow_pickle=True)
        singular_values = np.load('singular_values.npy')
        print("✓ Loaded reduced data, predictions, and singular values")
    except Exception as e:
        print(f"\nMissing files: {e}")
        print("\nPlease add these lines to your code and re-run main.py:")
        print("\nIn svd.py (fit method):")
        print("  np.save('singular_values.npy', self.singular_values)")
        print("\nIn main.py (after SVD transform):")
        print("  np.save('X_train_reduced.npy', X_train_reduced)")
        print("  np.save('X_test_reduced.npy', X_test_reduced)")
        print("\nIn main.py (after prediction):")
        print("  np.save('y_pred.npy', y_pred)")
        return
    
    viz = CleanVisualizer()
    
    # Generate 5 plots
    class_names = sorted(np.unique(np.concatenate([y_train, y_test])))
    
    print("\n" + "="*60)
    viz.plot_1_singular_value_spectrum(singular_values)
    
    print("\n" + "="*60)
    viz.plot_2_confusion_matrix(y_test, y_pred, class_names)
    
    print("\n" + "="*60)
    viz.plot_3_misclassification_analysis(y_test, y_pred, class_names)
    
    print("\n" + "="*60)
    viz.plot_4_class_performance(y_test, y_pred, class_names)
    
    print("\n" + "="*60)
    viz.plot_5_dimension_comparison(X_train, X_train_reduced)
    
    print("\n" + "="*60)
    print(f"All 5 plots saved in '{viz.results_dir}/'")
    print("="*60)

if __name__ == "__main__":
    main()