import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# 1. Tạo tập dữ liệu giả lập 50 điểm
np.random.seed(42)
x = np.sort(np.random.uniform(-5, 5, 50))
noise = np.random.normal(0, 0.5, size=50)
y = 2 * x + np.sin(x) + noise

X = x.reshape(-1, 1)

# Tập x_test mịn để vẽ đường dự đoán
X_test = np.linspace(-5, 5, 500).reshape(-1, 1)

# 2. Huấn luyện DecisionTreeRegressor với max_depth lần lượt là 2, 5, 10
depths = [2, 5, 10]
plt.figure(figsize=(12, 6))

# Vẽ dữ liệu thực tế
plt.scatter(X, y, color='black', label='Dữ liệu thực tế (có nhiễu)', s=30)

colors = ['blue', 'green', 'red']

for depth, color in zip(depths, colors):
    reg = DecisionTreeRegressor(max_depth=depth, random_state=42)
    reg.fit(X, y)
    y_pred = reg.predict(X_test)
    
    plt.plot(X_test, y_pred, color=color, label=f'max_depth = {depth}', linewidth=2)

plt.xlabel('x')
plt.ylabel('y')
plt.title('CART Hồi quy với các độ sâu max_depth khác nhau')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
