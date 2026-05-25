import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 1. Đọc danh sách file từ test_split.txt theo đúng thứ tự Pytorch DataLoader đã nạp
image_names = []
labels_text = []
with open('test_split.txt', 'r') as f:
    for line in f.readlines():
        parts = line.split()
        if len(parts) >= 3:
            filename = parts[1].split('/')[-1].split('\\')[-1]
            image_names.append(filename)
            labels_text.append(parts[2])

# 2. Đọc ma trận xếp hạng từ file .npz
npz_path = r'D:\tranh\Documents\Study\DataMining\X-MIR-Reproduction\results\covid_densenet121_seed_0_epoch_1_ckpt.npz'
data = np.load(npz_path)
dists = data['dists']
labels = np.array(data['labels']).flatten()

# Kiểm tra độ lệch thứ tự (Safeguard)
if len(image_names) != len(labels):
    print("CẢNH BÁO: Số lượng ảnh trong txt không khớp với npz!")

# 3. Truy tìm ảnh Covid-19 (Label 2.0)
covid_indices = np.where(labels == 2.0)[0]

print("=== ĐANG TRUY QUÉT TÌM ẢNH COVID-19 ĐỂ VISUALIZE ===")
# Chúng ta sẽ thử quét 3 bệnh nhân Covid đầu tiên
for q_idx in covid_indices[:3]: 
    
    # Sắp xếp khoảng cách tăng dần để tìm Top gần nhất
    # Vị trí [0] luôn là chính bức ảnh đó (khoảng cách = 0), nên ta lấy Top từ [1] đến [4]
    ranks = np.argsort(dists[:, q_idx])
    top_indices = ranks[1:4] 

    print(f"\n[QUERY IMAGE] File: {image_names[q_idx]} | Nhãn gốc: {labels_text[q_idx]}")
    
    for i, res_idx in enumerate(top_indices):
        is_correct = "ĐÚNG (GREEN)" if labels[res_idx] == labels[q_idx] else "SAI (RED)"
        print(f"  -> Top {i+1}: {image_names[res_idx]} | Nhãn hệ thống dự đoán: {labels_text[res_idx]} | Đánh giá: {is_correct}")

print("\n(Hãy copy kết quả này gửi lại cho AI để cấu hình Visualizer)")