import numpy as np

# Khai báo dữ liệu
w = np.array([1, 2, -10])
x = np.array([3, 4, 1])
y_true = -1

# 1. Tính w^T * x
score = np.dot(w, x)

# 2. Xác định nhãn dự đoán y_hat
y_pred = 1 if score >= 0 else -1

# 3. Kiểm tra phân lớp sai
is_misclassified = (y_pred != y_true)

print("=== BÀI 3.27: PERCEPTRON CHECK 1 ===")
print(f"1. w^T * x = {score}")
print(f"2. Nhãn dự đoán (y_hat) = {y_pred}")
print(f"3. Nhãn thực tế (y) = {y_true}")
print(f"   => Điểm dữ liệu bị phân lớp sai? {is_misclassified}")
