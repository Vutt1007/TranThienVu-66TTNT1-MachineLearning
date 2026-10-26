from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Load tập dữ liệu
X, y = load_digits(return_X_y=True)

# 1. Huấn luyện Random Forest 500 cây với oob_score=True
rf = RandomForestClassifier(n_estimators=500, oob_score=True, random_state=42)
rf.fit(X, y)

# 2. Đánh giá OOB Score
oob_score = rf.oob_score_

# 3. Đánh giá bằng Cross-Validation 5-fold
cv_scores = cross_val_score(rf, X, y, cv=5)

print(f"OOB Score: {oob_score:.4f}")
print(f"5-Fold CV Scores: {cv_scores}")
print(f"5-Fold CV Mean Score: {cv_scores.mean():.4f}")
