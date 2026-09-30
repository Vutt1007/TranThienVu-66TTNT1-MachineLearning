import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Huấn luyện mô hình Perceptron.
        X: numpy array shape (n_samples, n_features)
        y: numpy array shape (n_samples,) với nhãn {-1, 1} hoặc {0, 1}
        """
        n_samples, n_features = X.shape

        # Đảm bảo nhãn chuyển về {-1, 1}
        y_transformed = np.where(y <= 0, -1, 1)

        # Khởi tạo trọng số và bias bằng 0
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Vòng lặp huấn luyện
        for _ in range(self.n_iters):
            errors = 0
            for idx, x_i in enumerate(X):
                # Tính z = w^T * x + b
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.sign(linear_output)
                # Đảm bảo trường hợp = 0 được tính là 1
                if y_predicted == 0:
                    y_predicted = 1

                # Nếu phân lớp sai: y_i * (w^T * x_i + b) <= 0
                if y_transformed[idx] * linear_output <= 0:
                    update = self.lr * y_transformed[idx]
                    self.weights += update * x_i
                    self.bias += update
                    errors += 1

            # Dừng sớm nếu tất cả các điểm đều phân lớp đúng (dữ liệu phân tách tuyến tính)
            if errors == 0:
                break

    def predict(self, X):
        """
        Dự báo nhãn cho dữ liệu mới.
        Trả về nhãn {-1, 1} (hoặc có thể map lại {0, 1}).
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)

# --- Kiểm tra hoạt động của Class ---
if __name__ == "__main__":
    # Dữ liệu thử nghiệm cổng AND
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([-1, -1, -1, 1])

    p = Perceptron(learning_rate=0.1, n_iters=10)
    p.fit(X_train, y_train)

    print("Trọng số (Weights):", p.weights)
    print("Độ lệch (Bias):", p.bias)
    print("Dự đoán:", p.predict(X_train))
