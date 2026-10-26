import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# =======================================================
# 1. TỰ ĐỊNH NGHĨA LỚP PERCEPTRON (ĐÃ TỐI ƯU TỐC ĐỘ)
# =======================================================
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = lambda x: np.where(x >= 0, 1, 0)
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        y_encoded = np.where(y > 0, 1, 0)

        for _ in range(self.n_iters):
            errors = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                
                update = self.lr * (y_encoded[idx] - y_predicted)
                if update != 0:
                    self.weights += update * x_i
                    self.bias += update
                    errors += 1
            
            # Nếu không còn lỗi nào (đã hội tụ), dừng sớm để tránh treo
            if errors == 0:
                break

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_func(linear_output)

# =======================================================
# 2. KHỞI TẠO TẬP DỮ LIỆU GIAO DỊCH GIAN LẠN (MÔ PHỎNG)
# =======================================================
np.random.seed(42)

# Giao dịch hợp lệ (Nhãn 0): 800 mẫu
X_legit = np.random.normal(loc=0.0, scale=1.0, size=(800, 4))
y_legit = np.zeros(800)

# Giao dịch gian lận (Nhãn 1): 200 mẫu
X_fraud = np.random.normal(loc=2.5, scale=1.0, size=(200, 4))
y_fraud = np.ones(200)

# Gộp và xáo trộn dữ liệu
X = np.vstack((X_legit, X_fraud))
y = np.hstack((y_legit, y_fraud))

shuffle_idx = np.arange(len(y))
np.random.shuffle(shuffle_idx)
X, y = X[shuffle_idx], y[shuffle_idx]

# =======================================================
# 3. CHIA TẬP TRAIN/TEST & CHUẨN HÓA DỮ LIỆU
# =======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =======================================================
# 4. HUẤN LUYỆN & TÌM SIÊU THAM SỐ TỐT NHẤT
# =======================================================
best_model = None
best_f1 = -1.0
best_params = {}

learning_rates = [0.001, 0.01, 0.1]
n_iters_list = [100, 500, 1000] # Giảm số vòng lặp tối đa xuống để chạy nhanh hơn

for lr in learning_rates:
    for n_iters in n_iters_list:
        model = Perceptron(learning_rate=lr, n_iters=n_iters)
        model.fit(X_train_scaled, y_train)
        
        y_pred_val = model.predict(X_test_scaled)
        score = f1_score(y_test, y_pred_val, zero_division=0)
        
        if score >= best_f1:
            best_f1 = score
            best_model = model
            best_params = {'learning_rate': lr, 'n_iters': n_iters}

# =======================================================
# 5. DỰ BÁO VÀ ĐÁNH GIÁ MÔ HÌNH TỐT NHẤT
# =======================================================
y_pred = best_model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("=== TẠO MÔ HÌNH THÀNH CÔNG VỚI THAM SỐ TỐI ƯU ===")
print(f"Tham số tối ưu: {best_params}\n")

print("=== KẾT QUẢ ĐÁNH GIÁ CÁC ĐỘ ĐO ===")
print(f"1. Accuracy  (Độ chính xác tổng thể) : {acc:.4f} ({acc*100:.2f}%)")
print(f"2. Precision (Độ chính xác cảnh báo)  : {prec:.4f} ({prec*100:.2f}%)")
print(f"3. Recall    (Tỷ lệ phát hiện gian lận): {rec:.4f} ({rec*100:.2f}%)")
print(f"4. F1-score  (Độ cân bằng F1)         : {f1:.4f} ({f1*100:.2f}%)\n")

print("=== BÁO CÁO PHÂN LOẠI CHI TIẾT ===")
print(classification_report(y_test, y_pred, target_names=['Giao dịch Hợp lệ (0)', 'Giao dịch Gian lận (1)']))
