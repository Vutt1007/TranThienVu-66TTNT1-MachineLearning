import math
import pandas as pd

# 1. Hàm tính Entropy của tập dữ liệu
# Mặc định use_log2=False để tính log_e cho ra kết quả trùng khớp chính xác với Slide bài giảng (Slide 13-14).
def entropy(data, target_col="buys_computer", use_log2=False):
    total_count = len(data)
    if total_count == 0:
        return 0.0

    counts = data[target_col].value_counts()
    ent = 0.0
    for count in counts:
        p = count / total_count
        if p > 0:
            ent -= p * (math.log2(p) if use_log2 else math.log(p))
    return ent

# 2. Hàm tính Entropy có điều kiện H(x, S)
def conditional_entropy(data, feature, target_col="buys_computer", use_log2=False):
    total_count = len(data)
    cond_ent = 0.0

    for value in data[feature].unique():
        sub_data = data[data[feature] == value]
        prob = len(sub_data) / total_count
        cond_ent += prob * entropy(sub_data, target_col, use_log2)

    return cond_ent

# 3. Hàm tính Information Gain G(x, S) = H(S) - H(x, S)
def information_gain(data, feature, target_col="buys_computer", use_log2=False):
    return entropy(data, target_col, use_log2) - conditional_entropy(data, feature, target_col, use_log2)

# 4. Cấu trúc Nút của Cây Quyết Định
class TreeNode:
    def __init__(self, feature=None, value=None):
        self.feature = feature  # Thuộc tính kiểm tra tại nút
        self.children = {}     # Các nhánh con {giá_trị_thuộc_tính: TreeNode}
        self.value = value      # Nhãn kết luận nếu là Nút Lá

    def is_leaf(self):
        return self.value is not None

# 5. Thuật toán tạo cây ID3 theo đệ quy
def build_id3_tree(data, features, target_col="buys_computer", use_log2=False):
    target_values = data[target_col].unique()

    # Điều kiện dừng 1: Tất cả mẫu thuộc cùng 1 lớp -> Nút lá
    if len(target_values) == 1:
        return TreeNode(value=target_values[0])

    # Điều kiện dừng 2: Hết thuộc tính để phân chia -> Nhãn đa số
    if len(features) == 0:
        majority_class = data[target_col].mode()[0]
        return TreeNode(value=majority_class)

    # Tìm thuộc tính x* có Information Gain lớn nhất (H(x, S) nhỏ nhất)
    best_feature = None
    best_gain = -1.0

    print(f"\n--- Đang xét nút hiện tại (Số lượng mẫu: {len(data)}) ---")
    current_entropy = entropy(data, target_col, use_log2)
    print(f"Entropy tập hiện tại H(S): {current_entropy:.4f}")

    for feat in features:
        gain = information_gain(data, feat, target_col, use_log2)
        cond_ent = conditional_entropy(data, feat, target_col, use_log2)
        print(f"Thuộc tính: {feat:<15} | H({feat}, S) = {cond_ent:.4f} | Gain = {gain:.4f}")

        if gain > best_gain:
            best_gain = gain
            best_feature = feat

    print(f"==> Thuộc tính được chọn: [{best_feature}] (Gain = {best_gain:.4f})")

    # Tạo nút mới với thuộc tính tốt nhất vừa chọn
    root = TreeNode(feature=best_feature)
    remaining_features = [f for f in features if f != best_feature]

    # Phân chia dữ liệu vào các nhánh con
    for val in data[best_feature].unique():
        sub_data = data[data[best_feature] == val]
        
        # Nếu tập con rỗng -> Gán nhãn đa số của tập cha
        if len(sub_data) == 0:
            majority_class = data[target_col].mode()[0]
            root.children[val] = TreeNode(value=majority_class)
        else:
            root.children[val] = build_id3_tree(sub_data, remaining_features, target_col, use_log2)

    return root

# 6. Hàm in cây quyết định
def print_tree(node, depth=0, branch_val=""):
    indent = "   " * depth
    if node.is_leaf():
        print(f"{indent}└── [{branch_val}] ==> Kết luận (buys_computer): {node.value}")
    else:
        if depth == 0:
            print(f"Root: [{node.feature}]")
        else:
            print(f"{indent}└── [{branch_val}] --> [Kiểm tra: {node.feature}]")
        for val, child in node.children.items():
            print_tree(child, depth + 1, val)

# =========================================================================
# CHƯƠNG TRÌNH CHÍNH (Bài tập Slide 31)
# =========================================================================
if __name__ == "__main__":
    dataset = {
        "age": [
            "<=30", "<=30", "31...40", ">40", ">40", ">40", "31...40",
            "<=30", "<=30", ">40", "<=30", "31...40", "31...40", ">40"
        ],
        "income": [
            "high", "high", "high", "medium", "low", "low", "low",
            "medium", "low", "medium", "medium", "medium", "high", "medium"
        ],
        "student": [
            "no", "no", "no", "no", "yes", "yes", "yes",
            "no", "yes", "yes", "yes", "no", "yes", "no"
        ],
        "credit_rating": [
            "fair", "excellent", "fair", "fair", "fair", "excellent", "excellent",
            "fair", "fair", "fair", "excellent", "excellent", "fair", "excellent"
        ],
        "buys_computer": [
            "no", "no", "yes", "yes", "yes", "no", "yes",
            "no", "yes", "yes", "yes", "yes", "yes", "no"
        ]
    }

    df = pd.DataFrame(dataset)
    feature_columns = ["age", "income", "student", "credit_rating"]

    print("=== TÍNH TOÁN BÀI TẬP BẰNG ID3 (Log e - Giống Slide bài giảng) ===")
    decision_tree = build_id3_tree(df, feature_columns, use_log2=False)

    print("\n" + "=" * 50)
    print("CÂY QUYẾT ĐỊNH HOÀN CHỈNH:")
    print("=" * 50)
    print_tree(decision_tree)
