import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# ĐỊNH NGHĨA LỚP PERCEPTRON (Thêm phần này)
# ==========================================
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Khởi tạo trọng số và bias bằng 0
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Huấn luyện theo thuật toán Perceptron
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Tính giá trị dự báo: y = sign(w^T * x + b)
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.where(linear_output >= 0, 1, -1)

                # Cập nhật trọng số nếu dự báo sai: y_i * (w^T * x_i + b) <= 0
                if y[idx] * linear_output <= 0:
                    update = self.lr * y[idx]
                    self.weights += update * x_i
                    self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)


# ==========================================
# LUỒNG XỬ LÝ DỮ LIỆU VÀ ĐÁNH GIÁ
# ==========================================

# 1. Tạo dữ liệu mô phỏng bài toán phân lớp nhị phân
X, y = make_classification(
    n_samples=1000, 
    n_features=10, 
    n_informative=8, 
    n_redundant=2,
    weights=[0.9, 0.1], # Mất cân bằng dữ liệu (10% gian lận)
    random_state=42
)

# Chuyển nhãn từ {0, 1} về {-1, 1} cho Perceptron
y_perceptron = np.where(y == 0, -1, 1)

# 2. Chia tập Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_perceptron, test_size=0.3, random_state=42, stratify=y_perceptron
)

# 3. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Huấn luyện mô hình Perceptron
model = Perceptron(learning_rate=0.01, n_iters=1000)
model.fit(X_train_scaled, y_train)

# 5. Dự báo trên tập Test
y_pred = model.predict(X_test_scaled)

# 6. Tính toán các độ đo đánh giá (zero_division=0 tránh lỗi nếu không dự đoán được lớp positive nào)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

print("="*40)
print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH PERCEPTRON")
print("="*40)
print(f"Accuracy  (Độ chính xác tổng thể) : {acc * 100:.2f}%")
print(f"Precision (Độ xác thực)           : {prec * 100:.2f}%")
print(f"Recall    (Độ nhạy / Thu hồi)      : {rec * 100:.2f}%")
print(f"F1-Score  (Điểm dung hòa F1)       : {f1 * 100:.2f}%")
print("="*40)
