import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, rand_score

class MyKMeans:
    def __init__(self, K=3, max_iters=100, random_state=42):
        self.K = K
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def fit(self, X):
        np.random.seed(self.random_state)
        # 1. Khởi tạo tâm cụm ngẫu nhiên từ các điểm dữ liệu
        idx = np.random.choice(X.shape[0], self.K, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iters):
            # 2. Gán từng điểm vào tâm cụm gần nhất (Khoảng cách Euclidean)
            # distances shape: (N, K)
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            new_labels = np.argmin(distances, axis=1)

            # 3. Cập nhật tâm cụm mới bằng trung bình cộng các điểm trong cụm
            new_centroids = np.array([
                X[new_labels == k].mean(axis=0) if np.sum(new_labels == k) > 0 else self.centroids[k]
                for k in range(self.K)
            ])

            # 4. Kiểm tra hội tụ (nếu tâm cụm không thay đổi thì dừng)
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids
            self.labels_ = new_labels

        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

# --- Thực thi Bài 2 ---
# 1. Tải dữ liệu Iris & chỉ lấy 2 cột đầu tiên (sepal length, sepal width)
iris = load_iris()
X = iris.data[:, :2] # Sepal length, Sepal width
y_true = iris.target

# 2. Huấn luyện MyKMeans với K=3
kmeans = MyKMeans(K=3, max_iters=100, random_state=42)
kmeans.fit(X)
y_pred = kmeans.labels_

# 3. Tính chỉ số Rand Index
ri = rand_score(y_true, y_pred)
ari = adjusted_rand_score(y_true, y_pred)
print(f"Rand Index (RI): {ri:.4f}")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")

# 4. Vẽ biểu đồ phân tán so sánh
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Nhãn thực tế
axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', edgecolor='k', s=50)
axes[0].set_title("Nhãn thực tế (Ground Truth)")
axes[0].set_xlabel("Sepal Length")
axes[0].set_ylabel("Sepal Width")

# Kết quả K-Means tự cài đặt
axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', edgecolor='k', s=50)
axes[1].scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='X', s=200, label='Tâm cụm')
axes[1].set_title(f"Phân cụm MyKMeans (RI = {ri:.2f})")
axes[1].set_xlabel("Sepal Length")
axes[1].set_ylabel("Sepal Width")
axes[1].legend()

plt.tight_layout()
plt.show()
