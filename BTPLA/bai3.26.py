import numpy as np

# 1. Định nghĩa hàm f(x) và đạo hàm f'(x)
def f(x):
    return x**2 - 4*x + 5

def df(x):
    return 2*x - 4

# Tham số khởi tạo
x_k = 5.0
eta = 0.2
n_steps = 4

print("=== BÀI 3.26: GRADIENT DESCENT ===")
print(f"Bước 0: x^(0) = {x_k:.4f}, f(x^(0)) = {f(x_k):.4f}")

for k in range(1, n_steps + 1):
    grad = df(x_k)
    x_k = x_k - eta * grad
    print(f"Bước {k}: f'(x^({k-1})) = {grad:.4f} | x^({k}) = {x_k:.4f} | f(x^({k})) = {f(x_k):.4f}")

# Kiểm tra điểm hội tụ lý thuyết (x* = 2)
print(f"\nĐiểm cực tiểu lý thuyết: x* = 2.0, f(x*) = {f(2.0)}")
