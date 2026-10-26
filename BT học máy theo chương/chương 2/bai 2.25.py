import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Tạo tập dữ liệu điểm thi học sinh (500 học sinh)
np.random.seed(42)
n_samples = 500

hours_studied = np.random.uniform(1, 10, n_samples)
attendance = np.random.uniform(50, 100, n_samples)
sleep_hours = np.random.uniform(4, 9, n_samples)
previous_scores = np.random.uniform(40, 100, n_samples)
tutoring_sessions = np.random.randint(0, 6, n_samples)

# Công thức quan hệ thực tế có thêm nhiễu ngẫu nhiên
final_score = (
    0.35 * previous_scores +
    3.20 * hours_studied +
    0.25 * attendance +
    1.20 * sleep_hours +
    1.50 * tutoring_sessions +
    np.random.normal(0, 3.5, n_samples)
)
final_score = np.clip(final_score, 0, 100) # Giới hạn thang điểm [0, 100]

df = pd.DataFrame({
    'Hours_Studied': hours_studied,
    'Attendance': attendance,
    'Sleep_Hours': sleep_hours,
    'Previous_Scores': previous_scores,
    'Tutoring_Sessions': tutoring_sessions,
    'Final_Exam_Score': final_score
})

# 2. Chia dữ liệu Train/Test (80% / 20%)
X = df.drop(columns=['Final_Exam_Score'])
y = df['Final_Exam_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Định nghĩa hàm tính các độ đo R2, NSE, MAE, RMSE
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # Nash-Sutcliffe Efficiency (NSE)
    nse = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return r2, nse, mae, rmse

# 4. Huấn luyện và so sánh các mô hình
models = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge Regression (alpha=1.0)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Lasso Regression (alpha=0.1)": make_pipeline(StandardScaler(), Lasso(alpha=0.1)),
    "Polynomial Regression (degree=2)": make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), Ridge(alpha=10.0))
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2, nse, mae, rmse = calculate_metrics(y_test, y_pred)
    results.append({
        "Mô hình": name,
        "R2": round(r2, 4),
        "NSE": round(nse, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4)
    })

# 5. Hiển thị kết quả
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
