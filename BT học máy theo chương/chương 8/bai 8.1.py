import numpy as np
import matplotlib.pyplot as plt

# 1. Khai báo mẫu dữ liệu gốc
X = np.array([58, 70, 82, 95, 63, 77, 88, 69, 73, 85])
n_bootstraps = 2000
n_samples = len(X)

# Cố định seed để kết quả có thể lặp lại
np.random.seed(42)

# 2. Tạo 2000 mẫu bootstrap và tính trung bình mỗi mẫu
bootstrap_means = np.empty(n_bootstraps)
for i in range(n_bootstraps):
    # Lấy mẫu có hoàn lại (replace=True)
    sample = np.random.choice(X, size=n_samples, replace=True)
    bootstrap_means[i] = np.mean(sample)

# 3. Xác định khoảng tin cậy 95% bằng phương pháp Percentile
ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)

print(f"Trung bình mẫu gốc: {np.mean(X):.2f}")
print(f"Khoảng tin cậy 95%: [{ci_lower:.2f}, {ci_upper:.2f}]")

# 4. Vẽ histogram của phân phối bootstrap
plt.figure(figsize=(8, 5))
plt.hist(bootstrap_means, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(ci_lower, color='red', linestyle='--', linewidth=2, label=f'CI Dưới (2.5%): {ci_lower:.2f}')
plt.axvline(ci_upper, color='red', linestyle='--', linewidth=2, label=f'CI Trên (97.5%): {ci_upper:.2f}')
plt.axvline(np.mean(X), color='green', linestyle='-', linewidth=2, label=f'Trung bình gốc: {np.mean(X):.2f}')

plt.title('Phân phối Bootstrap của giá trị trung bình')
plt.xlabel('Giá trị trung bình')
plt.ylabel('Tần suất')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
