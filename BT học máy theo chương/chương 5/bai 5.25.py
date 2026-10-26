import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# 1. TỰ TẠO DỮ LIỆU KHÁCH HÀNG VÀ LƯU VÀO FILE CSV
data = {
    'Tuoi': [25, 45, 35, 50, 23, 40, 60, 18, 30, 55, 28, 42, 38, 22, 48],
    'ThuNhap': [15, 40, 25, 50, 12, 35, 45, 8, 20, 60, 18, 38, 28, 10, 42],  # Đơn vị: Triệu VNĐ/tháng
    'DiemTieuDung': [30, 80, 50, 90, 20, 75, 85, 10, 40, 95, 35, 70, 60, 15, 80], # Biến mục tiêu số (Hồi quy)
    'KhachHangTiemNang': [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]           # Biến mục tiêu phân loại (Phân lớp: 0/1)
}

df_customers = pd.DataFrame(data)
csv_filename = 'customer_data.csv'
df_customers.to_csv(csv_filename, index=False)
print(f"Đã tạo file CSV mẫu thành công: {csv_filename}")

# 2. ĐỌC DỮ LIỆU TỪ FILE CSV
df = pd.read_csv(csv_filename)
features = ['Tuoi', 'ThuNhap']
X = df[features]

# ==============================================================================
# PHẦN 1: CÂY PHÂN LỚP (DECISION TREE CLASSIFIER)
# ==============================================================================
print("\n" + "="*50)
print("1. CÂY PHÂN LỚP (Dự đoán Khách hàng tiềm năng)")
print("="*50)

y_cls = df['KhachHangTiemNang']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cls, test_size=0.3, random_state=42)

# Cắt tỉa cây bằng cách thay đổi 'min_samples_leaf'
leaf_values = [1, 2, 3]

for leaf in leaf_values:
    clf = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=42)
    clf.fit(X_train_c, y_train_c)
    acc = accuracy_score(y_test_c, clf.predict(X_test_c))
    print(f"-> Cây phân lớp với min_samples_leaf={leaf} | Độ chính xác (Accuracy): {acc:.4f}")

# Huấn luyện mô hình chuẩn để trực quan hóa và in tầm quan trọng đặc trưng
clf_final = DecisionTreeClassifier(min_samples_leaf=2, random_state=42)
clf_final.fit(X_train_c, y_train_c)

print("\nTầm quan trọng của các đặc trưng (Classification Feature Importances):")
for feat, imp in zip(features, clf_final.feature_importances_):
    print(f" - {feat}: {imp:.4f}")

# Trực quan hóa cây phân lớp và lưu thành file ảnh
plt.figure(figsize=(10, 6))
plot_tree(clf_final, feature_names=features, class_names=['Không', 'Có'], filled=True)
plt.title("Cây Phân Lớp (Classification Tree)")
classifier_img = 'classification_tree.png'
plt.savefig(classifier_img, bbox_inches='tight')
plt.close()
print(f"Đã lưu biểu đồ cây phân lớp vào file: {classifier_img}")

# ==============================================================================
# PHẦN 2: CÂY HỒI QUY (DECISION TREE REGRESSOR)
# ==============================================================================
print("\n" + "="*50)
print("2. CÂY HỒI QUY (Dự đoán Điểm tiêu dùng - Biến mục tiêu số)")
print("="*50)

y_reg = df['DiemTieuDung']
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.3, random_state=42)

# Cắt tỉa cây bằng cách thay đổi 'min_samples_leaf'
for leaf in leaf_values:
    reg = DecisionTreeRegressor(min_samples_leaf=leaf, random_state=42)
    reg.fit(X_train_r, y_train_r)
    mse = mean_squared_error(y_test_r, reg.predict(X_test_r))
    print(f"-> Cây hồi quy với min_samples_leaf={leaf} | Sai số MSE: {mse:.4f}")

# Huấn luyện mô hình chuẩn để trực quan hóa và in tầm quan trọng đặc trưng
reg_final = DecisionTreeRegressor(min_samples_leaf=2, random_state=42)
reg_final.fit(X_train_r, y_train_r)

print("\nTầm quan trọng của các đặc trưng (Regression Feature Importances):")
for feat, imp in zip(features, reg_final.feature_importances_):
    print(f" - {feat}: {imp:.4f}")

# Trực quan hóa cây hồi quy và lưu thành file ảnh
plt.figure(figsize=(10, 6))
plot_tree(reg_final, feature_names=features, filled=True)
plt.title("Cây Hồi Quy (Regression Tree)")
regressor_img = 'regression_tree.png'
plt.savefig(regressor_img, bbox_inches='tight')
plt.close()
print(f"Đã lưu biểu đồ cây hồi quy vào file: {regressor_img}")
