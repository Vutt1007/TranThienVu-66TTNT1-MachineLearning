import numpy as np

# Tập dữ liệu 1D
X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([-1, -1, -1, 1, 1, 1, 1, 1])

N = len(X)
w = np.ones(N) / N  # Khởi tạo trọng số bằng nhau = 1/N

T = 3
alphas = []
stumps = []

print("--- Quá trình huấn luyện AdaBoost ---")
for t in range(T):
    best_err = float('inf')
    best_stump = {}
    
    # Tạo danh sách các ngưỡng phân chia giữa các điểm dữ liệu
    thresholds = np.concatenate([[0.5], (X[:-1] + X[1:]) / 2.0, [8.5]])
    
    # Duyệt qua từng ngưỡng và hướng phân loại (< hoặc >)
    for thresh in thresholds:
        for inequality in ['lt', 'gt']:
            if inequality == 'lt':
                preds = np.where(X < thresh, -1, 1)
            else:
                preds = np.where(X > thresh, -1, 1)
            
            # Tính lỗi có trọng số (weighted error)
            err = np.sum(w[preds != y])
            
            if err < best_err:
                best_err = err
                best_stump = {'thresh': thresh, 'ineq': inequality, 'preds': preds}
    
    # Tránh lỗi chia cho 0 nếu err = 0
    err_safe = max(best_err, 1e-10)
    
    # Tính trọng số alpha cho bộ ước lượng t
    alpha = 0.5 * np.log((1 - err_safe) / err_safe)
    
    # Cập nhật trọng số của các mẫu dữ liệu
    w = w * np.exp(-alpha * y * best_stump['preds'])
    w /= np.sum(w)  # Chuẩn hóa trọng số
    
    alphas.append(alpha)
    stumps.append(best_stump)
    
    print(f"Vòng {t+1}: Ngưỡng = {best_stump['thresh']}, Hướng = '{best_stump['ineq']}', "
          f"Lỗi (Error) = {best_err:.4f}, Alpha = {alpha:.4f}")

# Dự đoán của Ensemble
final_preds_raw = np.zeros(N)
for alpha, stump in zip(alphas, stumps):
    final_preds_raw += alpha * stump['preds']

final_preds = np.sign(final_preds_raw)

print("\n--- Kết quả Ensemble ---")
print(f"Giá trị dự đoán thô (raw): {final_preds_raw}")
print(f"Dự đoán lớp cuối cùng: {final_preds.astype(int)}")
print(f"Độ chính xác trên tập dữ liệu: {(final_preds == y).mean() * 100:.2f}%")
