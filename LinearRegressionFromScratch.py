import numpy as np

class LinearRegressionFromScratch:
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        
        self.lr = learning_rate      # Calculus: Step size
        self.iters = n_iterations    # Number of iterations
        self.weights = None          # Weights w (Vector in Linear Algebra)
        self.bias = None             # Bias b (Scalar)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Training process using Gradient Descent.
        """
        n_samples, n_features = X.shape # Extract matrix dimensions

        # 1. Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 2. Optimization loop (Gradient Descent)
        for _ in range(self.iters):
            # Prediction: y = Xw + b (Matrix multiplication - Linear Algebra)
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute derivatives (Calculus: Derivatives)
            # dw is the derivative of the Loss function with respect to w
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters (Optimization step)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Prediction function for new values
        return np.dot(X, self.weights) + self.bias