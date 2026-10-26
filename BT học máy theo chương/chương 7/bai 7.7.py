import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN as SklearnDBSCAN

class MyDBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        n_samples = X.shape[0]
        # Khởi tạo nhãn cho tất cả điểm: -1 biểu thị noise (nhiễu), 0 biểu thị chưa phân loại
        self.labels_ = np.full(n_samples, -1)
        
        # 1. Tính ma trận khoảng cách Euclidean giữa mọi cặp điểm (Shape: N x N)
        dist_matrix = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2)
        
        # 2. Tìm hàng xóm của từng điểm trong bán kính eps
        neighbors = [np.where(dist_matrix[i] <= self.eps)[0] for i in range(n_samples)]
        
        # 3. Xác định các điểm Core Point
        is_core = np.array([len(neigh) >= self.min_samples for neigh in neighbors])
        
        cluster_id = 0
        visited = np.zeros(n_samples, dtype=bool)

        for i in range(n_samples):
            if visited[i] or not is_core[i]:
                continue
            
            # Bắt đầu một cụm mới
            visited[i] = True
            self.labels_[i] = cluster_id
            
            # Hàng chờ để mở rộng vùng (Region Query Expansion)
            seeds = list(neighbors[i])
            idx = 0
            
            while idx < len(seeds):
                point = seeds[idx]
                idx += 1
                
                if not visited[point]:
                    visited[point] = True
                    self.labels_[point] = cluster_id
                    
                    # Nếu điểm hàng xóm cũng là Core point, thêm các hàng xóm của nó vào hàng chờ
                    if is_core[point]:
                        for neighbor_pt in neighbors[point]:
                            if neighbor_pt not in seeds:
                                seeds.append(neighbor_pt)
                else:
                    # Nếu điểm đã thăm nhưng đang mang nhãn nhiễu (-1), gán lại vào cụm này (Border Point)
                    if self.labels_[point] == -1:
                        self.labels_[point] = cluster_id
                        
            cluster_id += 1

        return self

# --- Thực thi & So sánh với sklearn ---
# Tạo dữ liệu make_blobs có thêm nhiễu
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)
np.random.seed(42)
noise = np.random.uniform(low=-10, high=10, size=(30, 2))
X_data = np.vstack([X, noise])

eps = 0.8
min_pts = 5

# MyDBSCAN tự cài
my_dbscan = MyDBSCAN(eps=eps, min_samples=min_pts)
my_dbscan.fit(X_data)

# Sklearn DBSCAN
sk_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_pts)
sk_labels = sk_dbscan.fit_predict(X_data)

# Kiểm tra mức độ khớp nhau giữa 2 thuật toán
is_equal = np.array_equal(my_dbscan.labels_, sk_labels)
print(f"Kết quả của MyDBSCAN khớp hoàn toàn với Sklearn DBSCAN: {is_equal}")

# Vẽ đồ thị so sánh
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X_data[:, 0], X_data[:, 1], c=my_dbscan.labels_, cmap='tab10', s=30)
axes[0].set_title("MyDBSCAN (Tự cài đặt)")

axes[1].scatter(X_data[:, 0], X_data[:, 1], c=sk_labels, cmap='tab10', s=30)
axes[1].set_title("sklearn.cluster.DBSCAN")

plt.tight_layout()
plt.show()
