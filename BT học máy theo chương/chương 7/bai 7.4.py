import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image

# 1. Tải ảnh mẫu (hoặc dùng ảnh bất kỳ qua PIL/OpenCV)
china = load_sample_image("china.jpg") # Ảnh mẫu kích thước (427, 640, 3)
china = np.array(china, dtype=np.float64) / 255.0 # Chuẩn hóa RGB về [0, 1]

w, h, d = china.shape
# Biến đổi ảnh 3D (chiều rộng, chiều cao, RGB) thành mảng 2D (số pixel, 3)
image_array = np.reshape(china, (w * h, d))

# 2. Áp dụng K-Means với K = 16
K = 16
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans.fit(image_array)

# Thay thế giá trị từng pixel bằng giá trị tâm cụm tương ứng
compressed_palette = kmeans.cluster_centers_
labels = kmeans.labels_
compressed_image = compressed_palette[labels].reshape(w, h, d)

# 3. Tính tỷ lệ nén (về mặt số màu)
# Ảnh gốc có thể chứa tới 256^3 = 16,777,216 màu phân biệt.
original_colors = len(np.unique(image_array.reshape(-1, 3), axis=0))
compressed_colors = K
color_compression_ratio = original_colors / compressed_colors

print(f"Số lượng màu ban đầu trong ảnh: {original_colors:,}")
print(f"Số lượng màu sau khi nén: {compressed_colors}")
print(f"Tỷ lệ giảm số lượng màu: {color_compression_ratio:.2f} lần")

# 4. Hiển thị ảnh gốc và ảnh sau nén
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(china)
axes[0].set_title(f"Ảnh gốc ({original_colors:,} màu)")
axes[0].axis('off')

axes[1].imshow(compressed_image)
axes[1].set_title(f"Ảnh nén (K = {K} màu)")
axes[1].axis('off')

plt.tight_layout()
plt.show()
