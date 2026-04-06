import os
import argparse
import numpy as np
import gc

from features import build_dataset, process_audio, standardize
from svd import ManualSVD
from classifier import ManualKNN
from metrics import calculate_metrics, print_confusion_matrix

def get_data_with_cache(data_dir, X_cache_file, y_cache_file):
    if os.path.exists(X_cache_file) and os.path.exists(y_cache_file):
        print(f"Loading cached features from {X_cache_file}...")
        X = np.load(X_cache_file)
        y = np.load(y_cache_file, allow_pickle=True)
    else:
        print(f"No cache found. Extracting from {data_dir}...")
        X, y = build_dataset(data_dir)
        print(f"Saving extracted features to {X_cache_file} to save time in the future...")
        np.save(X_cache_file, X)
        np.save(y_cache_file, y)
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Audio Classification with Linear SVD and kNN")
    parser.add_argument('--train_dir', type=str, default='data/train', help='Path to training data')
    parser.add_argument('--test_dir', type=str, default='data/test', help='Path to test data')
    parser.add_argument('--predict', type=str, default=None, help='Path to a single .wav file to predict')
    args = parser.parse_args()

    print("\n--- Phase 1: Feature Extraction ---")
    X_train_raw, y_train = get_data_with_cache(args.train_dir, 'X_train.npy', 'y_train.npy')
    X_train_norm, mean_train, std_train = standardize(X_train_raw)

    del X_train_raw
    gc.collect()

    print("\n--- Phase 2: Dimensionality Reduction (SVD) ---")
    svd = ManualSVD(k=25)
    svd.fit(X_train_norm)
    X_train_reduced = svd.transform(X_train_norm)

    del X_train_norm
    gc.collect()

    print("\n--- Phase 3: Assembly & Model Fitting ---")
    knn = ManualKNN(k=5)
    knn.fit(X_train_reduced, y_train)

    if args.predict:
        print(f"\nPredicting single file: {args.predict}")
        x_test_raw = process_audio(args.predict).reshape(1, -1)
        x_test_norm, _, _ = standardize(x_test_raw, mean=mean_train, std=std_train)
        x_test_reduced = svd.transform(x_test_norm)
        prediction = knn.predict(x_test_reduced)[0]
        print(f"\nPredicted Class: {prediction}")
        
    else:
        print("\nEvaluating on Test Set...")
        X_test_raw, y_test = get_data_with_cache(args.test_dir, 'X_test.npy', 'y_test.npy')
        
        if len(y_test) == 0:
            print("No test data found. Make sure sort_data.py was run successfully.")
            return
            
        X_test_norm, _, _ = standardize(X_test_raw, mean=mean_train, std=std_train)
        
        del X_test_raw
        gc.collect()
        
        X_test_reduced = svd.transform(X_test_norm)
        
        del X_test_norm
        gc.collect()
        
        y_pred = knn.predict(X_test_reduced, batch_size=500)
        
        print("\n--- Final Evaluation Metrics ---")
        results = calculate_metrics(y_test, y_pred)
        
        print(f"Overall Accuracy:   {results['Accuracy'] * 100:.2f}%")
        print(f"Weighted Precision: {results['Weighted_Precision']:.4f}")
        print(f"Weighted Recall:    {results['Weighted_Recall']:.4f}")
        print(f"Weighted F1-Score:  {results['Weighted_F1']:.4f}")
        
        print_confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    main()