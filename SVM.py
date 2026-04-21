import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class SVM:
    def __init__(self, epoch, lr, C, random_state=42):
        self.epoch = epoch
        self.lr = lr
        self.C = C
        self.random_state = random_state
        self.losses = []
    
    def predict_value(self, x: np.ndarray, w: np.ndarray, b: float):
        return (w.T @ x + b)
        
    def hinge_loss(self, C, values: np.ndarray, w: np.ndarray):
        return (0.5 * w.T @ w + C * np.mean(np.maximum(0, 1 - values)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, d = X.shape
        self.w = np.zeros((d,), dtype=np.float64)
        self.b = 0
        rng = np.random.default_rng(self.random_state)

        for i in tqdm(range(self.epoch)):
            idx = rng.permutation(N)
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            for j in range(N):
                y_hat = y_shuffled[j] * self.predict_value(X_shuffled[j], self.w, self.b)
                
                if y_hat >= 1:
                    gradient =  self.w
                    self.w = self.w - self.lr * gradient

                elif y_hat < 1:
                    gradient = self.w - self.C * y_shuffled[j] * X_shuffled[j]
                    self.w = self.w - self.lr * gradient
                    self.b = self.b + self.lr * self.C * y_shuffled[j]
                
            epoch_margins = y * (X @ self.w + self.b)
            
            loss = self.hinge_loss(self.C, epoch_margins, self.w)
            self.losses.append(loss)
        
    def predict(self, X):
        value =  X @ self.w + self.b
        y_pred = np.where(value >= 0, 1, -1)
        return y_pred
    
    def evaluate(self, y, y_pred) -> dict:
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)   
        f1 = f1_score(y, y_pred)

        print(f"accuracy: {acc}")
        print(f"precision: {prec}")
        print(f"recall: {rec}")
        print(f"f1_score: {f1}")