import numpy as np
from pathlib import Path

def calculate_retrieval_metrics(npz_file_path):
    if not Path(npz_file_path).exists():
        return None
        
    # Tải dữ liệu từ file .npz
    data = np.load(npz_file_path)
    dists = data['dists']
    labels = np.array(data['labels']).flatten()
    
    num_queries = len(labels)
    aps = []
    p1s = []
    p5s = []
    
    # Sắp xếp khoảng cách tăng dần cho từng Query
    # Vị trí [0] là chính nó (khoảng cách = 0), ta lấy từ vị trí [1:] trở đi
    sorted_ranks = np.argsort(dists, axis=0)
    
    for q in range(num_queries):
        query_label = labels[q]
        # Loại bỏ chính ảnh query ra khỏi danh sách kết quả truy xuất
        retrieved_indices = sorted_ranks[1:, q]
        retrieved_labels = labels[retrieved_indices]
        
        # 1. Tính Precision @ 1 và Precision @ 5
        p1s.append(1.0 if retrieved_labels[0] == query_label else 0.0)
        p5s.append(np.sum(retrieved_labels[:5] == query_label) / 5.0)
        
        # 2. Tính Average Precision (AP) cho từng query
        correct_mask = (retrieved_labels == query_label)
        if not np.any(correct_mask):
            aps.append(0.0)
            continue
            
        ranks = np.arange(1, len(retrieved_labels) + 1)
        precisions = np.cumsum(correct_mask) / ranks
        ap = np.sum(precisions * correct_mask) / np.sum(correct_mask)
        aps.append(ap)
        
    # Tính trung bình toàn bộ tập dữ liệu (Global Metrics)
    mAP = np.mean(aps) * 100
    mP1 = np.mean(p1s) * 100
    mP5 = np.mean(p5s) * 100
    
    return mAP, mP1, mP5

# Cấu hình đường dẫn tới 2 file .npz trên máy của bạn
covid_npz = r'D:\tranh\Documents\Study\DataMining\X-MIR-Reproduction\results\covid_densenet121_seed_0_epoch_1_ckpt.npz'
isic_npz = r'D:\tranh\Documents\Study\DataMining\X-MIR-Reproduction\results\isic_densenet121_embed_256_seed_0_epoch_20_ckpt.npz'

print("=====================================================================")
print("                      BẢNG 1: RETRIEVAL RESULTS                      ")
print("=====================================================================")
print(f"{'Dataset':<15} | {'Model':<20} | {'mAP ↑':<8} | {'P@1 ↑':<8} | {'P@5 ↑':<8}")
print("-" * 68)

# Chạy và in kết quả COVID-19
covid_res = calculate_retrieval_metrics(covid_npz)
if covid_res:
    print(f"{'COVID-19':<15} | {'DenseNet-121':<20} | {covid_res[0]:<8.1f} | {covid_res[1]:<8.1f} | {covid_res[2]:<8.1f}")
else:
    print(f"{'COVID-19':<15} | Không tìm thấy file .npz")

# Chạy và in kết quả ISIC 2017
isic_res = calculate_retrieval_metrics(isic_npz)
if isic_res:
    print(f"{'ISIC 2017':<15} | {'DenseNet-121':<20} | {isic_res[0]:<8.1f} | {isic_res[1]:<8.1f} | {isic_res[2]:<8.1f}")
else:
    print(f"{'ISIC 2017':<15} | Không tìm thấy file .npz")
print("=====================================================================")