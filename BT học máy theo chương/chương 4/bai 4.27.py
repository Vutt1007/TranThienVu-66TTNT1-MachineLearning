import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Định nghĩa độ đo NSE (Nash-Sutcliffe Efficiency)
def nash_sutcliffe_efficiency(y_true, y_pred):
    return 1.0 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

# 2. Tải tập dữ liệu Giá nhà California
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

# 3. Chia tập dữ liệu và Chuẩn hóa
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Xây dựng và huấn luyện Mạng Nơ-ron (MLPRegressor)
model = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    solver='adam',
    max_iter=100,
    random_state=42
)

print("=== BÀI 4.27: ĐANG HUẤN LUYỆN MÔ HÌNH HỒI QUY GIÁ NHÀ ===")
model.fit(X_train_scaled, y_train)

# 5. Dự báo và Đánh giá theo các độ đo yêu cầu
y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
nse = nash_sutcliffe_efficiency(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ---")
print(f"Hệ số xác định (R2):  {r2:.4f}")
print(f"Hiệu quả Nash (NSE):  {nse:.4f}")
print(f"Sai số MAE:            {mae:.4f}")
print(f"Sai số RMSE:           {rmse:.4f}")

# 6. NHẬP DỮ LIỆU TỪ BÀN PHÍM VÀ DỰ BÁO GIÁ NHÀ
print("\n" + "="*50)
print("NHẬP THÔNG TIN CĂN NHÀ ĐỂ DỰ BÁO GIÁ")
print("(Gợi ý: Nhấn Enter để sử dụng giá trị mặc định/trung bình)")
print("="*50)

user_input = []
for i, name in enumerate(feature_names):
    mean_val = np.mean(X[:, i])
    val_str = input(f"Nhập {name} (Giá trị trung bình ~ {mean_val:.2f}): ")
    
    # Nếu người dùng không nhập gì mà nhấn Enter thì lấy giá trị trung bình
    if val_str.strip() == "":
        val = mean_val
    else:
        val = float(val_str)
        
    user_input.append(val)

# Chuyển dữ liệu vừa nhập thành mảng 2D và chuẩn hóa bằng scaler đã fit
user_data = np.array([user_input])
user_data_scaled = scaler.transform(user_data)

# Dự báo giá nhà
predicted_price = model.predict(user_data_scaled)[0]

# Đơn vị tính của tập dữ liệu California Housing là $100,000
price_in_usd = predicted_price * 100000

print("\n" + "*"*50)
print(f"GIÁ NHÀ DỰ BÁO: {predicted_price:.4f} (khoảng ${price_in_usd:,.2f})")
print("*"*50)
