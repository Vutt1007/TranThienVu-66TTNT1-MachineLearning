import math
import pandas as pd


# 1. Hàm tính Entropy cơ bản
def entropy(target_series):
    total = len(target_series)
    if total == 0:
        return 0.0
    counts = target_series.value_counts()
    ent = 0.0
    for count in counts:
        p = count / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent


# 2. Hàm tìm điểm cắt tối ưu (Best Threshold) cho thuộc tính liên tục
def find_best_split_continuous(df, feature, target_col="Rủi ro tín dụng"):
    sorted_df = df[[feature, target_col]].sort_values(by=feature)
    unique_vals = sorted_df[feature].unique()

    if len(unique_vals) <= 1:
        return None, 0.0

    # Các điểm cắt tiềm năng là trung điểm giữa 2 giá trị liên tiếp
    thresholds = [
        (unique_vals[i] + unique_vals[i + 1]) / 2.0
        for i in range(len(unique_vals) - 1)
    ]

    base_entropy = entropy(df[target_col])
    best_threshold = None
    best_gain = -1.0

    for t in thresholds:
        left = df[df[feature] <= t][target_col]
        right = df[df[feature] > t][target_col]

        cond_entropy = (len(left) / len(df)) * entropy(left) + (
            len(right) / len(df)
        ) * entropy(right)
        gain = base_entropy - cond_entropy

        if gain > best_gain:
            best_gain = gain
            best_threshold = t

    return best_threshold, best_gain


# 3. Hàm tính Information Gain cho thuộc tính rời rạc
def information_gain_categorical(df, feature, target_col="Rủi ro tín dụng"):
    base_entropy = entropy(df[target_col])
    total = len(df)
    cond_entropy = 0.0

    for val in df[feature].unique():
        sub_df = df[df[feature] == val]
        cond_entropy += (len(sub_df) / total) * entropy(sub_df[target_col])

    return base_entropy - cond_entropy


# 4. Cấu trúc Nút cây quyết định
class TreeNode:

    def __init__(
        self,
        feature=None,
        threshold=None,
        is_continuous=False,
        value=None,
    ):
        self.feature = feature
        self.threshold = threshold
        self.is_continuous = is_continuous
        self.children = {}  # Lưu nhánh con
        self.value = value  # Nhãn lớp nếu là nút lá

    def is_leaf(self):
        return self.value is not None


# 5. Thuật toán ID3 đệ quy
def build_id3_tree(
    df, features, continuous_features, target_col="Rủi ro tín dụng"
):
    target_vals = df[target_col].unique()

    # Điều kiện dừng 1: Tinh khiết (Tất cả mẫu cùng 1 lớp)
    if len(target_vals) == 1:
        return TreeNode(value=target_vals[0])

    # Điều kiện dừng 2: Hết thuộc tính để phân chia -> Chọn nhãn chiếm đa số
    if len(features) == 0:
        majority_class = df[target_col].mode()[0]
        return TreeNode(value=majority_class)

    best_feature = None
    best_gain = -1.0
    best_threshold = None

    print(
        f"\n--- TÍNH INFORMATION GAIN TẠI NÚT HIỆN TẠI (Số mẫu: {len(df)}) ---"
    )
    print(f"Entropy tập hiện tại H(S) = {entropy(df[target_col]):.4f}")

    # Lặp qua tất cả thuộc tính để chọn thuộc tính có Gain cao nhất
    for feat in features:
        if feat in continuous_features:
            t, gain = find_best_split_continuous(df, feat, target_col)
            print(
                f"Thuộc tính liên tục [{feat:<12}]: Threshold = {t} | Gain = {gain:.4f}"
            )
            if gain > best_gain:
                best_gain = gain
                best_feature = feat
                best_threshold = t
        else:
            gain = information_gain_categorical(df, feat, target_col)
            print(f"Thuộc tính rời rạc [{feat:<12}]: Gain = {gain:.4f}")
            if gain > best_gain:
                best_gain = gain
                best_feature = feat
                best_threshold = None

    print(
        f"==> Thuộc tính tốt nhất được chọn: [{best_feature}] (Gain = {best_gain:.4f})"
    )

    # Phân chia nhánh đệ quy
    if best_feature in continuous_features:
        node = TreeNode(
            feature=best_feature,
            threshold=best_threshold,
            is_continuous=True,
        )

        # Nhánh <= Threshold
        left_df = df[df[best_feature] <= best_threshold]
        if not left_df.empty:
            node.children[f"<= {best_threshold}"] = build_id3_tree(
                left_df, features, continuous_features, target_col
            )

        # Nhánh > Threshold
        right_df = df[df[best_feature] > best_threshold]
        if not right_df.empty:
            node.children[f"> {best_threshold}"] = build_id3_tree(
                right_df, features, continuous_features, target_col
            )
    else:
        node = TreeNode(
            feature=best_feature, is_continuous=False
        )
        remaining_features = [f for f in features if f != best_feature]

        for val in df[best_feature].unique():
            sub_df = df[df[best_feature] == val]
            node.children[val] = build_id3_tree(
                sub_df, remaining_features, continuous_features, target_col
            )

    return node


# 6. Hàm in cây quyết định
def print_tree(node, depth=0, branch_info=""):
    indent = "   " * depth
    if node.is_leaf():
        print(f"{indent}└── [{branch_info}] ==> Rủi ro tín dụng: {node.value}")
    else:
        if depth == 0:
            print(f"Gốc (Root): [{node.feature}]")
        else:
            print(f"{indent}└── [{branch_info}] --> Kiểm tra [{node.feature}]")
        for val, child in node.children.items():
            print_tree(child, depth + 1, val)


# =========================================================================
# KHỞI TẠO DỮ LIỆU BÀI TẬP TỪ BẢNG
# =========================================================================
if __name__ == "__main__":
    data = {
        "Độ tuổi": [25, 40, 35, 27, 31, 36, 48, 26, 33, 29, 38, 44, 42, 28, 30],
        "Hôn nhân": [
            "Độc thân",
            "Đã kết hôn",
            "Từng ly hôn",
            "Đã kết hôn",
            "Độc thân",
            "Đã kết hôn",
            "Độc thân",
            "Đã kết hôn",
            "Từng ly hôn",
            "Độc thân",
            "Đã kết hôn",
            "Độc thân",
            "Đã kết hôn",
            "Độc thân",
            "Đã kết hôn",
        ],
        "Sở hữu BĐS": [
            "Ở cùng bố mẹ",
            "Nhà sở hữu",
            "Nhà thuê",
            "Ở cùng bố mẹ",
            "Nhà thuê",
            "Nhà sở hữu",
            "Nhà thuê",
            "Nhà sở hữu",
            "Ở cùng bố mẹ",
            "Nhà thuê",
            "Nhà sở hữu",
            "Nhà sở hữu",
            "Nhà sở hữu",
            "Nhà thuê",
            "Ở cùng bố mẹ",
        ],
        "Thu nhập": [
            7000000,
            18000000,
            12000000,
            9000000,
            6000000,
            8000000,
            7000000,
            8000000,
            5000000,
            10000000,
            15000000,
            14000000,
            10000000,
            7000000,
            6000000,
        ],
        "Rủi ro tín dụng": [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
    }

    df = pd.DataFrame(data)

    features = ["Độ tuổi", "Hôn nhân", "Sở hữu BĐS", "Thu nhập"]
    continuous_features = ["Độ tuổi", "Thu nhập"]

    print("================ TÍNH TOÁN ID3 ================")
    id3_tree = build_id3_tree(df, features, continuous_features)

    print("\n================ CÂY QUYẾT ĐỊNH HOÀN CHỈNH ================")
    print_tree(id3_tree)
