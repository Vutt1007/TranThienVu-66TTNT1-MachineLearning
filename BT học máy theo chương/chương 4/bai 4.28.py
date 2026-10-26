import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1. Tạo tập dữ liệu giả lập về Điểm thi (Chuyên cần, BTVN, Giữa kỳ)
np.random.seed(42)
n_samples = 1000

diem_chuyen_can = np.random.uniform(0, 10, n_samples)
diem_btvn = np.random.uniform(0, 10, n_samples)
diem_giua_ky = np.random.uniform(0, 10, n_samples)

# Điều kiện: Đỗ (1) nếu điểm tổng hợp >= 5.0, ngược lại Trượt (0)
score_formula = 0.2 * diem_chuyen_can + 0.3 * diem_btvn + 0.5 * diem_giua_ky
label = (score_formula >= 5.0).astype(int)

X = np.column_stack((diem_chuyen_can, diem_btvn, diem_giua_ky))
y = label

# 2. Chia tập dữ liệu và Chuẩn hóa
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Xây dựng và huấn luyện Mạng Nơ-ron Phân lớp (MLPClassifier)
model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=42
)

print("=== BÀI 4.28: ĐANG HUẤN LUYỆN MÔ HÌNH PHÂN LỚP ĐIỂM THI ===")
model.fit(X_train_scaled, y_train)

# 4. Dự báo và Đánh giá theo Accuracy, Precision, Recall, F1-Score
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"Độ chính xác (Accuracy):   {acc:.4f}")
print(f"Precision:                {prec:.4f}")
print(f"Recall:                   {rec:.4f}")
print(f"F1-Score:                 {f1:.4f}")
print("\nBáo cáo chi tiết:")
print(classification_report(y_test, y_pred, target_names=['Trượt (0)', 'Đỗ (1)']))
