import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        """
        Khởi tạo các tham số cơ bản cho Perceptron.
        - learning_rate (float): Tốc độ học (eta).
        - n_iters (int): Số vòng lặp huấn luyện (epochs).
        """
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def _unit_step_func(self, x):
        """
        Hàm kích hoạt Step Function: Trả về 1 nếu x >= 0, ngược lại trả về 0.
        """
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        """
        Hàm huấn luyện mô hình Perceptron.
        - X (numpy.ndarray): Tập dữ liệu huấn luyện kích thước (n_samples, n_features).
        - y (numpy.ndarray): Nhãn thực tế kích thước (n_samples,).
        """
        n_samples, n_features = X.shape

        # 1. Khởi tạo trọng số (weights) bằng 0 (hoặc số ngẫu nhiên nhỏ) và bias bằng 0
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Đảm bảo y thuộc nhãn {0, 1}
        y_encoded = np.where(y > 0, 1, 0)

        # 2. Vòng lặp huấn luyện qua số lần n_iters
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Tính tổng có trọng số: z = w * x + b
                linear_output = np.dot(x_i, self.weights) + self.bias
                
                # Áp dụng hàm kích hoạt để đưa ra dự báo
                y_predicted = self.activation_func(linear_output)

                # Tính độ chênh lệch (Error)
                update = self.lr * (y_encoded[idx] - y_predicted)

                # Cập nhật trọng số và bias
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """
        Hàm dự báo nhãn cho dữ liệu mới.
        - X (numpy.ndarray): Kích thước (n_samples, n_features).
        - Return: Mảng chứa nhãn dự báo (0 hoặc 1).
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted
