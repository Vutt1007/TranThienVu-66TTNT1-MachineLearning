import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 1. Tạo tập dữ liệu make_blobs: 500 mẫu, 5 tâm thực
X, y_true = make_blobs(n_samples=500, centers=5, cluster_std=1.0, random_state=42)

# 2. Tính Inertia cho K từ 1 đến 10
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# 3. Vẽ đường cong Elbow
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.title("Phương pháp Elbow tìm K tối ưu")
plt.xlabel("Số lượng cụm (K)")
plt.ylabel("Inertia (Tổng bình phương khoảng cách)")
plt.xticks(K_range)
plt.grid(True)
plt.show()

# 4. Nhận xét & So sánh với Ground Truth:
# Quan sát đồ thị, điểm gập (elbow) xuất hiện rõ nhất tại K = 5.
# Tại K = 5, tốc độ giảm của inertia bắt đầu chậm lại đáng kể.
# Kết quả này hoàn toàn khớp với ground truth (5 tâm thực tế khi khởi tạo make_blobs).
