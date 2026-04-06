import numpy as np
from collections import Counter
import time

class ManualKNN:
    def __init__(self, k=5):
        """
        Initializes the custom kNN classifier.
        """
        self.k = k
        self.X_train = None
        self.y_train = None
        self.X_train_norms_sq = None

    def fit(self, X_train, y_train):
        """
        Stores the projected training data and precomputes the squared Euclidean norms.
        """
        self.X_train = X_train
        self.y_train = np.array(y_train)
        self.X_train_norms_sq = np.sum(X_train**2, axis=1)

    def predict(self, X_test, batch_size=500):
        """
        Predicts labels for test data using the expanded Euclidean distance formula.
        Processes in batches to prevent ArrayMemoryError.
        """
        predictions = []
        start_time = time.time()
        
        num_test_samples = X_test.shape[0]
        
        for i in range(0, num_test_samples, batch_size):
            X_batch = X_test[i:i + batch_size]
            X_batch_norms_sq = np.sum(X_batch**2, axis=1)

            dist_sq = X_batch_norms_sq[:, np.newaxis] + self.X_train_norms_sq - 2 * np.dot(X_batch, self.X_train.T)
            
            dist_sq = np.maximum(dist_sq, 0)
            distances = np.sqrt(dist_sq)
            
            for dist_vector in distances:
                k_indices = np.argsort(dist_vector)[:self.k]
                k_nearest_labels = self.y_train[k_indices]

                vote = Counter(k_nearest_labels).most_common(1)[0][0]
                predictions.append(vote)
                
            print(f"Predicted {min(i + batch_size, num_test_samples)}/{num_test_samples} samples...")
            
        end_time = time.time()
        avg_time = (end_time - start_time) / num_test_samples if num_test_samples > 0 else 0
        print(f"kNN prediction completed. Average prediction time: {avg_time:.4f} seconds per sample.")
        
        return np.array(predictions)