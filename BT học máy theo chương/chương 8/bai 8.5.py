import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# 1. Tạo tập dữ liệu
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=5, 
    random_state=42
)

# 2. Chia train/test 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Khởi tạo các mô hình Ensemble với 100 bộ ước lượng
models = {
    'BaggingClassifier': BaggingClassifier(
        estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42
    ),
    'RandomForestClassifier': RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    'AdaBoostClassifier': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, algorithm='SAMME', random_state=42
    ),
    'GradientBoostingClassifier': GradientBoostingClassifier(
        n_estimators=100, random_state=42
    )
}

# 4. Thực hành huấn luyện và so sánh
print(f"{'Phương pháp':28s} | {'Accuracy':10s} | {'Thời gian (s)':12s}")
print("-" * 58)

for name, model in models.items():
    t_start = time.time()
    model.fit(X_train, y_train)
    t_train = time.time() - t_start
    
    acc = model.score(X_test, y_test)
    print(f"{name:28s} | {acc:<10.4f} | {t_train:<12.4f}")
