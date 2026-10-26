import numpy as np

class NeuralNetworkFromScratch:
    def __init__(self, layer_sizes, task_type='regression'):
        """
        layer_sizes: List chứa số lượng neuron cho từng tầng [input_dim, h1, h2, ..., output_dim]
        task_type: 'regression' hoặc 'classification'
        """
        self.layer_sizes = layer_sizes
        self.task_type = task_type
        self.weights = []
        self.biases = []
        
        # Khởi tạo trọng số (He Initialization)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return (z > 0).astype(float)

    def forward(self, X):
        self.activations = [X]
        self.zs = []
        
        curr_activation = X
        # Các tầng ẩn (sử dụng ReLU)
        for i in range(len(self.weights) - 1):
            z = np.dot(curr_activation, self.weights[i]) + self.biases[i]
            curr_activation = self._relu(z)
            self.zs.append(z)
            self.activations.append(curr_activation)
            
        # Tầng đầu ra
        z_out = np.dot(curr_activation, self.weights[-1]) + self.biases[-1]
        self.zs.append(z_out)
        
        if self.task_type == 'classification':
            a_out = self._sigmoid(z_out)
        else:
            a_out = z_out
            
        self.activations.append(a_out)
        return a_out

    def fit(self, X, y, epochs=1000, lr=0.01):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).reshape(-1, 1)
        m = X.shape[0]

        for epoch in range(epochs):
            # 1. Lan truyền tiến (Forward)
            y_pred = self.forward(X)

            # 2. Tính Loss và Delta ở tầng đầu ra
            if self.task_type == 'classification':
                loss = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
                delta = y_pred - y
            else:
                loss = np.mean((y_pred - y) ** 2)
                delta = 2 * (y_pred - y) / m

            # 3. Lan truyền ngược (Backward Pass)
            dW_list = []
            db_list = []

            for i in range(len(self.weights) - 1, -1, -1):
                dW = np.dot(self.activations[i].T, delta)
                db = np.sum(delta, axis=0, keepdims=True)
                
                dW_list.insert(0, dW)
                db_list.insert(0, db)

                if i > 0:
                    delta = np.dot(delta, self.weights[i].T) * self._relu_derivative(self.zs[i-1])

            # 4. Cập nhật Trọng số
            for i in range(len(self.weights)):
                self.weights[i] -= lr * dW_list[i]
                self.biases[i] -= lr * db_list[i]

            if (epoch + 1) % 200 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        pred = self.forward(X)
        if self.task_type == 'classification':
            return (pred >= 0.5).astype(int)
        return pred


# --- Chạy thử nghiệm Bài 4.26 ---
if __name__ == "__main__":
    print("=== BÀI 4.26: HUẤN LUYỆN MẠNG NƠ-RON TỪ SCRATCH ===")
    np.random.seed(42)
    X_dummy = np.random.randn(100, 5)
    y_dummy = np.random.randn(100, 1)

    # Khởi tạo mô hình: 5 input -> 2 tầng ẩn (16, 8) -> 1 output
    model = NeuralNetworkFromScratch(layer_sizes=[5, 16, 8, 1], task_type='regression')
    model.fit(X_dummy, y_dummy, epochs=1000, lr=0.01)
    print("Đã hoàn thành huấn luyện bài 4.26 thành công!")
