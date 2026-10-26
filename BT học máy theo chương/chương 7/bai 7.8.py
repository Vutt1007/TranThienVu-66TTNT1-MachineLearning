import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN

# 1. Tạo tập dữ liệu make_circles
X, _ = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)

# 2. Định nghĩa danh sách các cặp (eps, minPts)
params = [
    (0.2, 5),
    (0.5, 5),
    (1.0, 5),
    (0.5, 10)
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, (eps, min_pts) in enumerate(params):
    dbscan = DBSCAN(eps=eps, min_samples=min_pts)
    labels = dbscan.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=20)
    axes[i].set_title(f"$\epsilon = {eps}, minPts = {min_pts}$\nSố cụm: {n_clusters} | Nhiễu: {n_noise}")
    axes[i].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
