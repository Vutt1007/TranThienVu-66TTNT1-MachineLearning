import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Tạo dữ liệu mô phỏng bài toán phân lớp nhị phân (vd: Giao dịch gian lận)
# Hoặc thay thế bằng: df = pd.read_csv('creditcard.csv')
X, y = make_classification(
    n_samples=1000, 
    n_features=10, 
    n_informative=8, 
    n_redundant=2,
    weights=[0.9, 0.1], # Giả lập mất cân bằng dữ liệu (10% gian lận)
    random_state=42
)

# Chuyển nhãn về {-1, 1} cho Perceptron
y_perceptron = np.where(y == 0, -1, 1)

# 2. Chia tập Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_perceptron, test_size=0.3, random_state=42, stratify=y_perceptron
)

# 3. Chuẩn hóa dữ liệu (Rất quan trọng đối với Perceptron/Gradient Descent)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Huấn luyện mô hình Perceptron (Sử dụng class đã viết ở bài 3.29)
model = Perceptron(learning_rate=0.01, n_iters=1000)
model.fit(X_train_scaled, y_train)

# 5. Dự báo trên tập Test
y_pred = model.predict(X_test_scaled)

# 6. Tính toán các độ đo danh giá
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=1)
rec = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

print("="*40)
print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH PERCEPTRON")
print("="*40)
print(f"Accuracy  (Độ chính xác tổng thể) : {acc * 100:.2f}%")
print(f"Precision (Độ xác thực)          : {prec * 100:.2f}%")
print(f"Recall    (Độ nhạy / Thu hồi)     : {rec * 100:.2f}%")
print(f"F1-Score  (Điểm dung hòa F1)      : {f1 * 100:.2f}%")
print("="*40)
