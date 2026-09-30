import numpy as np
import pandas as pd

# Thay thế io.StringIO bằng Dictionary trực tiếp
data = {
    "So_phong": [
        2, 3, 4, 1, 3, 5, 2, 4, 3, 2, 6, 3, 4, 1, 5, 3, 2, 4, 5, 3, 2, 4, 6, 3, 5
    ],
    "Dien_tich_m2": [
        65, 110, 215, 52, 145, 310, 80, 180, 125, 75, 275, 150, 200, 45, 260,
        105, 88, 220, 290, 135, 95, 170, 320, 115, 240
    ],
    "Khoang_cach_km": [
        4.5, 8.2, 2.5, 15.0, 5.8, 18.2, 1.8, 9.4, 11.2, 3.2, 4.1, 14.5, 6.3, 8.0,
        12.0, 2.1, 13.8, 17.5, 3.0, 7.6, 10.4, 1.5, 8.8, 16.0, 5.2
    ],
    "Gia_nha_trieu_VND": [
        3900, 6200, 12800, 2300, 8100, 14500, 5400, 9800, 6900, 4800, 16200,
        7800, 11200, 2700, 13100, 6800, 4600, 10900, 17500, 7400, 5100, 10500,
        18200, 5900, 13800
    ]
}

df = pd.DataFrame(data)

X = df[["So_phong", "Dien_tich_m2", "Khoang_cach_km"]].values
y = df["Gia_nha_trieu_VND"].values

np.random.seed(42)

indices = np.random.permutation(len(X))

train_size = 17
val_size = 4

train_idx = indices[:train_size]
val_idx = indices[train_size : train_size + val_size]
test_idx = indices[train_size + val_size :]

X_train = X[train_idx]
y_train = y[train_idx]

X_val = X[val_idx]
y_val = y[val_idx]

X_test = X[test_idx]
y_test = y[test_idx]

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std


def tao_ma_tran(X, bac):
    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]

    ma_tran = [np.ones(len(X)), x1, x2, x3]

    if bac >= 2:
        ma_tran += [
            x1**2,
            x2**2,
            x3**2,
            x1 * x2,
            x1 * x3,
            x2 * x3,
        ]

    if bac >= 3:
        ma_tran += [
            x1**3,
            x2**3,
            x3**3,
            x1**2 * x2,
            x1**2 * x3,
            x2**2 * x1,
            x2**2 * x3,
            x3**2 * x1,
            x3**2 * x2,
            x1 * x2 * x3,
        ]

    if bac >= 4:
        ma_tran += [
            x1**4,
            x2**4,
            x3**4,
            x1**3 * x2,
            x1**3 * x3,
            x2**3 * x1,
            x2**3 * x3,
            x3**3 * x1,
            x3**3 * x2,
        ]

    if bac >= 5:
        ma_tran += [x1**5, x2**5, x3**5]

    if bac >= 6:
        ma_tran += [x1**6, x2**6, x3**6]

    if bac >= 7:
        ma_tran += [x1**7, x2**7, x3**7]

    if bac >= 8:
        ma_tran += [x1**8, x2**8, x3**8]

    return np.column_stack(ma_tran)


def tim_W(X, y):
    W = np.linalg.pinv(X.T @ X) @ X.T @ y
    return W


def du_doan(X, W):
    return X @ W


def MSE(y_that, y_du_doan):
    return np.mean((y_that - y_du_doan) ** 2)


X_train_1 = tao_ma_tran(X_train, 1)
X_val_1 = tao_ma_tran(X_val, 1)

W_1 = tim_W(X_train_1, y_train)

train_pred_1 = du_doan(X_train_1, W_1)
val_pred_1 = du_doan(X_val_1, W_1)

train_mse_1 = MSE(y_train, train_pred_1)
val_mse_1 = MSE(y_val, val_pred_1)

X_train_8 = tao_ma_tran(X_train, 8)
X_val_8 = tao_ma_tran(X_val, 8)

W_8 = tim_W(X_train_8, y_train)

train_pred_8 = du_doan(X_train_8, W_8)
val_pred_8 = du_doan(X_val_8, W_8)

train_mse_8 = MSE(y_train, train_pred_8)
val_mse_8 = MSE(y_val, val_pred_8)

print("====================================")
print("MO HINH BAN DAU")
print("====================================")
print("Bac 1")
print("Train MSE:", round(train_mse_1, 2))
print("Validation MSE:", round(val_mse_1, 2))

print("\n====================================")
print("MO HINH BI OVERFIT")
print("====================================")
print("Bac 8")
print("Train MSE:", round(train_mse_8, 2))
print("Validation MSE:", round(val_mse_8, 2))

candidate_degrees = [1, 2, 3]

best_degree = 1
best_mse = float("inf")

for degree in candidate_degrees:
    X_tr = tao_ma_tran(X_train, degree)
    X_val_temp = tao_ma_tran(X_val, degree)

    W = tim_W(X_tr, y_train)
    pred = du_doan(X_val_temp, W)
    error = MSE(y_val, pred)

    if error < best_mse:
        best_mse = error
        best_degree = degree

X_train_best = tao_ma_tran(X_train, best_degree)
X_test_best = tao_ma_tran(X_test, best_degree)

W_best = tim_W(X_train_best, y_train)

test_pred = du_doan(X_test_best, W_best)
test_mse = MSE(y_test, test_pred)

print("\n====================================")
print("SAU KHI SUA OVERFIT")
print("====================================")
print("Bac duoc chon:", best_degree)
print("Validation MSE:", round(best_mse, 2))
print("Test MSE:", round(test_mse, 2))

print("\n====================================")
print("MO HINH")
print("====================================")
print("So luong tham so:", len(W_best))
print("\nTrong so W:")

for i in range(len(W_best)):
    print("W[" + str(i) + "] =", W_best[i])

print("\n====================================")
print("DU DOAN NHA MOI")
print("====================================")

so_phong = float(input("Nhap so phong: "))
dien_tich = float(input("Nhap dien tich (m2): "))
khoang_cach = float(input("Nhap khoang cach (km): "))

nha_moi = np.array([[so_phong, dien_tich, khoang_cach]])
nha_moi = (nha_moi - mean) / std

X_new = tao_ma_tran(nha_moi, best_degree)
gia_du_doan = du_doan(X_new, W_best)

print("\n====================================")
print("KET QUA")
print("====================================")
print("So phong:", so_phong)
print("Dien tich:", dien_tich, "m2")
print("Khoang cach:", khoang_cach, "km")
print(
    "Gia nha du doan:",
    round(gia_du_doan[0], 2),
    "trieu VND",
)
