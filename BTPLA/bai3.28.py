import numpy as np

# Khai báo dữ liệu
w = np.array([-2, 1, 0])
x = np.array([2, 3, 1])
y_true = 1
eta = 1.0  # Learning rate mặc định

print("=== BÀI 3.28: PERCEPTRON UPDATE ===")

# 1. Kiểm tra phân lớp
score = np.dot(w, x)
y_pred = 1 if score >= 0 else -1
print(f"1. w^T * x ban đầu = {score}")
print(f"   Nhãn dự đoán = {y_pred}, Nhãn thực tế = {y_true}")

if y_pred != y_true:
    print("   => Mẫu BỊ PHÂN LỚP SAI!")
    
    # 2. Cập nhật trọng số: w_new = w + eta * y * x
    w_new = w + eta * y_true * x
    print(f"2. Vector trọng số mới w_new = {w_new}")
    
    # 3. Tính lại w^T * x sau cập nhật
    score_new = np.dot(w_new, x)
    print(f"3. Giá trị w_new^T * x sau cập nhật = {score_new}")
else:
    print("   => Mẫu phân lớp ĐÚNG, không cần cập nhật.")
