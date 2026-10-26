import math

# 1. Khởi tạo tập dữ liệu "Mua máy tính"
data = [
    {'Tuổi': 'Trẻ', 'Thu nhập': 'Cao', 'Sinh viên': 'Không', 'Đánh giá tín dụng': 'Tốt', 'Mua máy tính': 'Không'},
    {'Tuổi': 'Trẻ', 'Thu nhập': 'Cao', 'Sinh viên': 'Không', 'Đánh giá tín dụng': 'Xuất sắc', 'Mua máy tính': 'Không'},
    {'Tuổi': 'Trung niên', 'Thu nhập': 'Cao', 'Sinh viên': 'Không', 'Đánh giá tín dụng': 'Tốt', 'Mua máy tính': 'Có'},
    {'Tuổi': 'Già', 'Thu nhập': 'Trung bình', 'Sinh viên': 'Không', 'Đánh giá tín dụng': 'Tốt', 'Mua máy tính': 'Có'},
    {'Tuổi': 'Già', 'Thu nhập': 'Thấp', 'Sinh viên': 'Có', 'Đánh giá tín dụng': 'Tốt', 'Mua máy tính': 'Có'},
    {'Tuổi': 'Già', 'Thu nhập': 'Thấp', 'Sinh viên': 'Có', 'Đánh giá tín dụng': 'Xuất sắc', 'Mua máy tính': 'Không'},
    {'Tuổi': 'Trung niên', 'Thu nhập': 'Thấp', 'Sinh viên': 'Có', 'Đánh giá tín dụng': 'Xuất sắc', 'Mua máy tính': 'Có'},
    {'Tuổi': 'Trẻ', 'Thu nhập': 'Trung bình', 'Sinh viên': 'Không', 'Đánh giá tín dụng': 'Tốt', 'Mua máy tính': 'Không'},
    {'Tuổi': 'Trẻ', 'Thu nhập': 'Thấp', 'Sinh viên': 'Có', 'Đánh giá tín dụng': 'Tốt', 'Mua máy tính': 'Có'},
    {'Tuổi': 'Già', 'Thu nhập': 'Trung bình', 'Sinh viên': 'Có', 'Đánh giá tín dụng': 'Tốt', 'Mua máy tính': 'Có'},
    {'Tuổi': 'Trẻ', 'Thu nhập': 'Trung bình', 'Sinh viên': 'Có', 'Đánh giá tín dụng': 'Xuất sắc', 'Mua máy tính': 'Có'},
    {'Tuổi': 'Trung niên', 'Thu nhập': 'Trung bình', 'Sinh viên': 'Không', 'Đánh giá tín dụng': 'Xuất sắc', 'Mua máy tính': 'Có'},
    {'Tuổi': 'Trung niên', 'Thu nhập': 'Cao', 'Sinh viên': 'Có', 'Đánh giá tín dụng': 'Tốt', 'Mua máy tính': 'Có'},
    {'Tuổi': 'Già', 'Thu nhập': 'Trung bình', 'Sinh viên': 'Không', 'Đánh giá tín dụng': 'Xuất sắc', 'Mua máy tính': 'Không'}
]

features = ['Tuổi', 'Thu nhập', 'Sinh viên', 'Đánh giá tín dụng']
target = 'Mua máy tính'

# 2. Hàm tính Entropy
def entropy(dataset, target_attribute):
    counts = {}
    for row in dataset:
        label = row[target_attribute]
        counts[label] = counts.get(label, 0) + 1
    
    ent = 0.0
    for label, count in counts.items():
        p = count / len(dataset)
        ent -= p * math.log2(p)
    return ent

# 3. Hàm tính Information Gain (Độ lợi thông tin)
def info_gain(dataset, feature, target_attribute):
    total_entropy = entropy(dataset, target_attribute)
    subset_counts = {}
    subsets = {}
    
    for row in dataset:
        val = row[feature]
        if val not in subsets:
            subsets[val] = []
        subsets[val].append(row)
        
    weighted_entropy = 0.0
    for val, subset in subsets.items():
        p = len(subset) / len(dataset)
        weighted_entropy += p * entropy(subset, target_attribute)
        
    return total_entropy - weighted_entropy

# 4. Hàm build_tree trả về cây dưới dạng dictionary
def build_tree(dataset, features, target_attribute):
    # Lấy danh sách các nhãn target
    labels = [row[target_attribute] for row in dataset]
    
    # Trường hợp 1: Tất cả mẫu cùng 1 nhãn -> Trả về nhãn đó
    if labels.count(labels[0]) == len(labels):
        return labels[0]
        
    # Trường hợp 2: Không còn thuộc tính phân chia -> Trả về nhãn xuất hiện nhiều nhất
    if not features:
        return max(set(labels), key=labels.count)
        
    # Chọn thuộc tính có Information Gain lớn nhất
    gains = {feat: info_gain(dataset, feat, target_attribute) for feat in features}
    best_feature = max(gains, key=gains.get)
    
    tree = {best_feature: {}}
    
    # Lấy các giá trị duy nhất của thuộc tính tốt nhất
    feature_values = set(row[best_feature] for row in dataset)
    
    for val in feature_values:
        sub_dataset = [row for row in dataset if row[best_feature] == val]
        remaining_features = [f for f in features if f != best_feature]
        tree[best_feature][val] = build_tree(sub_dataset, remaining_features, target_attribute)
        
    return tree

# 5. Hàm predict dự đoán nhãn cho mẫu mới
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    
    root_node = next(iter(tree))
    feature_value = sample.get(root_node)
    
    if feature_value in tree[root_node]:
        return predict(tree[root_node][feature_value], sample)
    else:
        return "Không xác định"

# --- CHẠY THỬ VÀ KIỂM TRA ---
tree = build_tree(data, features, target)
import pprint
print("=== CÂY QUYẾT ĐỊNH (DICTIONARY) ===")
pprint.pprint(tree)

# Kiểm tra trên các mẫu thử
test_samples = [
    {'Tuổi': 'Trẻ', 'Thu nhập': 'Trung bình', 'Sinh viên': 'Có', 'Đánh giá tín dụng': 'Tốt'},
    {'Tuổi': 'Già', 'Thu nhập': 'Thấp', 'Sinh viên': 'Không', 'Đánh giá tín dụng': 'Xuất sắc'}
]

print("\n=== KẾT QUẢ DỰ ĐOÁN MẪU THỬ ===")
for i, sample in enumerate(test_samples, 1):
    result = predict(tree, sample)
    print(f"Mẫu {i} {sample} -> Dự đoán 'Mua máy tính': {result}")
