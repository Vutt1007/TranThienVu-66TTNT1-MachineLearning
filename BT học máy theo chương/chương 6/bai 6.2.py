import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 1. Tải tập dữ liệu Iris và lấy 2 thuộc tính đầu tiên của 2 lớp Setosa & Versicolor
iris = load_iris()
X = iris.data[:100, :2]  # 100 mẫu đầu tương ứng setosa và versicolor
y = iris.target[:100]

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Lớp Logistic Regression tự cài đặt
class LogisticRegressionGD:
    def __init__(self, lr=0.01, epochs=200):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.losses = []

        for _ in range(self.epochs):
            z = np.dot(X, self.w) + self.b
            y_hat = sigmoid(z)
            
            # Loss cross-entropy
            loss = -np.mean(y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15))
            self.losses.append(loss)

            # Gradient
            dw = (1 / n_samples) * np.dot(X.T, (y_hat - y))
            db = (1 / n_samples) * np.sum(y_hat - y)

            # Update
            self.w -= self.lr * dw
            self.b -= self.lr * db

# 2. Thử nghiệm với 3 Learning Rate khác nhau
learning_rates = [0.01, 0.1, 1.0]
models = {}

plt.figure(figsize=(12, 4))

# Vẽ đồ thị Loss theo Epoch
plt.subplot(1, 2, 1)
for lr in learning_rates:
    model = LogisticRegressionGD(lr=lr, epochs=200)
    model.fit(X, y)
    models[lr] = model
    plt.plot(model.losses, label=f'lr = {lr}')

plt.xlabel('Epochs')
plt.ylabel('Loss (Cross-Entropy)')
plt.title('Loss theo Epoch cho các Learning Rate')
plt.legend()
plt.grid(True)

# 3. Vẽ đường biên phân lớp trên mặt phẳng 2D (chọn model với lr = 0.1)
plt.subplot(1, 2, 2)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Setosa (0)')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Versicolor (1)')

# Phương trình đường biên: w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b) / w2
best_model = models[0.1]
x1_vals = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])
x2_vals = -(best_model.w[0] * x1_vals + best_model.b) / best_model.w[1]

plt.plot(x1_vals, x2_vals, 'k--', label='Boundary (lr=0.1)')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Đường biên phân lớp 2D')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
