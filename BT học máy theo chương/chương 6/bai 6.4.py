import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# Tải dữ liệu Breast Cancer
data = load_breast_cancer()
X, y = data.data, data.target

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Cau a) Linear SVM voi C = 0.1, 1, 10 ---
print("--- CÂU A: LINEAR SVM ---")
for C in [0.1, 1.0, 10.0]:
    clf = SVC(kernel='linear', C=C, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    n_sv = np.sum(clf.n_support_)
    print(f"C = {C:4.1f} | So Support Vectors: {n_sv:3d} | Độ chính xác Test: {acc*100:.2f}%")

# Giải thích ảnh hưởng của C:
# - C nhỏ: Cho phép nhiều điểm vi phạm margin hơn (soft margin rộng), mô hình đơn giản hơn, số lượng Support Vectors nhiều hơn.
# - C lớn: Phạt nặng các điểm vi phạm margin (harder margin), cố gắng phân loại đúng tối đa tập huấn luyện, số lượng Support Vectors ít hơn nhưng dễ dẫn đến Overfitting.

# --- Cau b) RBF Kernel voi PCA giam chieu du lieu xuong 2D ---
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)

def plot_rbf_boundary(C, gamma, subplot_pos):
    clf = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
    clf.fit(X_train_2d, y_train)
    
    x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
    y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.subplot(2, 2, subplot_pos)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', s=20)
    plt.title(f"RBF Kernel: C={C}, gamma={gamma}")

plt.figure(figsize=(10, 8))
plot_rbf_boundary(C=0.1, gamma=0.01, subplot_pos=1)
plot_rbf_boundary(C=1.0, gamma=0.01, subplot_pos=2)
plot_rbf_boundary(C=10.0, gamma=0.1, subplot_pos=3)
plot_rbf_boundary(C=100.0, gamma=1.0, subplot_pos=4) # Rất dễ Overfitting

plt.tight_layout()
plt.show()
