import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Tạo tập dữ liệu 2D gồm 3 cụm dài, hẹp và lệch nhau
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
# Biến đổi tuyến tính để kéo dài và làm lệch các cụm
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X = np.dot(X, transformation)

# 2. Chạy K-means 20 lần độc lập cho mỗi phương pháp khởi tạo
n_runs = 20
inertias_random = []
inertias_kmeanspp = []

for seed in range(n_runs):
    # Khởi tạo ngẫu nhiên (random)
    km_rand = KMeans(n_clusters=3, init='random', n_init=1, max_iter=300, random_state=seed)
    km_rand.fit(X)
    inertias_random.append(km_rand.inertia_)
    
    # Khởi tạo K-means++
    km_pp = KMeans(n_clusters=3, init='k-means++', n_init=1, max_iter=300, random_state=seed)
    km_pp.fit(X)
    inertias_kmeanspp.append(km_pp.inertia_)

# 3. Vẽ Histogram so sánh
plt.figure(figsize=(10, 5))
plt.hist(inertias_random, bins=10, alpha=0.6, label='Random Init', color='orange', edgecolor='black')
plt.hist(inertias_kmeanspp, bins=10, alpha=0.6, label='K-Means++ Init', color='blue', edgecolor='black')
plt.title("So sánh phân bố Inertia giữa Random và K-Means++ (20 lần chạy)")
plt.xlabel("Inertia (Tổng bình phương khoảng cách)")
plt.ylabel("Tần suất (Số lần)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# --- Nhận xét ---
# 1. Khởi tạo K-means++ tập trung thu được các giá trị inertia nhỏ hơn và ổn định hơn hẳn so với Random.
# 2. Khởi tạo Random dễ bị vướng vào các cực trị địa phương (local minima) do chọn tâm cụm ngẫu nhiên ban đầu không hợp lý.
