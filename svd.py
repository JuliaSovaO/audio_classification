import numpy as np
import time

class ManualSVD:
    def __init__(self, k=25, epsilon=1e-10, max_iter=2000):
        self.k = k
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.V_k = None
        self.singular_values = None

    def _power_iteration(self, A):
        n = A.shape[0]
        
        np.random.seed(42)
        v = np.random.rand(n)
        v = v / np.linalg.norm(v)

        lambda_old = 0.0

        for _ in range(self.max_iter):
            Av = np.dot(A, v)
            v_new = Av / np.linalg.norm(Av)
            
            lambda_new = np.dot(v_new.T, np.dot(A, v_new))

            if np.abs(lambda_new - lambda_old) < self.epsilon:
                break
                
            lambda_old = lambda_new
            v = v_new

        return lambda_new, v

    def fit(self, X):
        print(f"Starting SVD computation. Features dimension: {X.shape[1]}")
        start_time = time.time()

        A = np.dot(X.T, X)
        eigenvalues = []
        eigenvectors = []

        for i in range(self.k):
            eigenval, eigenvec = self._power_iteration(A)
            
            eigenvalues.append(eigenval)
            eigenvectors.append(eigenvec)

            A = A - eigenval * np.outer(eigenvec, eigenvec)
            
            if (i + 1) % 5 == 0:
                print(f"Extracted {i + 1}/{self.k} components...")

        self.V_k = np.column_stack(eigenvectors)
        self.singular_values = np.sqrt(np.abs(eigenvalues))

        end_time = time.time()
        print(f"Manual SVD completed in {end_time - start_time:.2f} seconds.")
        
        return self

    def transform(self, X, batch_size=2000):
        if self.V_k is None:
            raise ValueError("The SVD model must be fitted before calling transform.")
        
        num_samples = X.shape[0]
        X_reduced = np.zeros((num_samples, self.k), dtype=X.dtype)
        
        for i in range(0, num_samples, batch_size):
            X_reduced[i:i + batch_size] = np.dot(X[i:i + batch_size], self.V_k)
            
        return X_reduced

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)