import numpy as np


# Định nghĩa đạo hàm f'(x)
def grad(x):
  return 2 * x


# Định nghĩa hàm chi phí f(x)
def cost(x):
  return x**2 - 2


# Thuật toán Gradient Descent
def myGD1(x0, eta):
  x = [x0]
  for it in range(100):
    x_new = x[-1] - eta * grad(x[-1])
    if abs(grad(x_new)) < 1e-3:  # Điều kiện dừng khi đạo hàm đủ nhỏ
      break
    x.append(x_new)
  return (x, it)
