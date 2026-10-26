import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# 1. Tải dữ liệu Iris
iris = load_iris()
X, y = iris.data, iris.target

# Chia tập dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Huấn luyện mô hình không giới hạn độ sâu (max_depth=None)
clf_full = DecisionTreeClassifier(criterion='gini', random_state=42)
clf_full.fit(X_train, y_train)

# Vẽ cây
plt.figure(figsize=(12, 8))
plot_tree(clf_full, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Cây quyết định CART (Không giới hạn độ sâu)")
plt.show()

print("--- Nhận xét về các nút (Cây không giới hạn) ---")
print("Cây phân nhánh cho đến khi các nút lá đạt độ tinh khiết tối đa (Gini = 0.0) hoặc không còn thuộc tính để chia.")
print("-" * 50)

# 3. Thay đổi max_depth thành 2 và 3, so sánh độ chính xác
depths = [None, 2, 3]

print(f"{'max_depth':<12} | {'Train Accuracy':<15} | {'Test Accuracy':<15}")
print("-" * 48)

for depth in depths:
    clf = DecisionTreeClassifier(criterion='gini', max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    
    depth_str = str(depth) if depth is not None else "None (Vô hạn)"
    print(f"{depth_str:<12} | {train_acc:<15.4f} | {test_acc:<15.4f}")

print("\n--- Nhận xét về hiện tượng quá khớp (Overfitting) ---")
print("1. Khi max_depth = None: Mô hình học quá chi tiết dữ liệu huấn luyện (Train Acc có thể đạt 100%), dễ dẫn đến Overfitting nếu dữ liệu nhiễu.")
print("2. Khi giới hạn max_depth = 2 hoặc 3: Cây được cắt tỉa bớt, giảm độ phức tạp, giúp mô hình tổng quát hóa tốt hơn trên tập Test mà độ chính xác vẫn giữ ở mức rất cao.")
