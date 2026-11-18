import os
import numpy as np

# Đường dẫn của bạn
isic_path = r'D:\tranh\Documents\Study\DataMining\X-MIR-Reproduction\data\ISIC2017\ISIC-2017_Test_v2_Data'
npz_path = r'D:\tranh\Documents\Study\DataMining\X-MIR-Reproduction\results\isic_densenet121_embed_256_seed_0_epoch_20_ckpt.npz'

# 1. Tải nhãn thực tế từ file .npz
data = np.load(npz_path)
labels = np.array(data['labels']).flatten()

# 2. Giả lập cách PyTorch đọc thư mục (Quét Alphabet)
image_names = []
simulated_labels = []

# Chỉ lấy các thư mục con và sắp xếp A->Z
classes = sorted([d for d in os.listdir(isic_path) if os.path.isdir(os.path.join(isic_path, d))])

for label_idx, cls in enumerate(classes):
    cls_path = os.path.join(isic_path, cls)
    # Lấy các file ảnh và sắp xếp A->Z
    files = sorted([f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    for f in files:
        image_names.append(f)
        simulated_labels.append(label_idx)

print(f"Tổng số ảnh quét được trong folder: {len(image_names)}")
print(f"Tổng số nhãn trong file .npz: {len(labels)}")

if len(image_names) == len(labels):
    # Kiểm tra xem thứ tự mảng nhãn giả lập có khớp 100% mảng nhãn của X-MIR không
    if np.array_equal(simulated_labels, labels):
        print("\n[SUCCESS] Tuyệt vời! Thứ tự đọc folder KHỚP HOÀN TOÀN với file .npz!")
        print("Mảng classes đang được map như sau:", dict(enumerate(classes)))
    else:
        print("\n[FAILED] Độ dài khớp nhưng thứ tự nhãn bị lệch. PyTorch không đọc theo Alphabet.")
else:
    print("\n[FAILED] Số lượng file quét được không khớp với file .npz.")