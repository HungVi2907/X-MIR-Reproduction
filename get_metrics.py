import numpy as np
import warnings
warnings.filterwarnings('ignore')

def compute_ap(ranks, nres):
    nimgranks = len(ranks)
    ap = 0
    recall_step = 1. / nres
    for j in np.arange(nimgranks):
        rank = ranks[j]
        precision_0 = 1. if rank == 0 else float(j) / rank
        precision_1 = float(j + 1) / (rank + 1)
        ap += (precision_0 + precision_1) * recall_step / 2.
    return ap

def compute_map(ranks, gnd, kappas=[]):
    mAP = 0.
    nq = len(gnd)
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.where(gnd == gnd[i])[0]
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue
            
        pos = np.arange(ranks.shape[0])[np.isin(ranks[:, i], qgnd)]
        ap = compute_ap(pos, len(qgnd))
        mAP = mAP + ap
        aps[i] = ap
        pos += 1
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j])
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]
    return mAP / (nq - nempty), aps, pr / (nq - nempty), prs

file_path = r'D:\tranh\Documents\Study\DataMining\X-MIR-Reproduction\results\covid_densenet121_seed_0_epoch_1_ckpt.npz'

print(f"Đang đọc dữ liệu từ: {file_path}")
data = np.load(file_path)
dists = data['dists']
labels = np.array(data['labels']).flatten()

unique_labels = np.unique(labels)

kappas = [1, 5]
# [SỬA LỖI QUAN TRỌNG]: Bỏ [::-1] để sắp xếp TĂNG DẦN (Khoảng cách nhỏ nhất lên top 1)
ranks = np.argsort(dists, axis=0) 

print("Đang tính toán lại mAP và P@K cho từng ảnh...")
global_mAP, aps, global_pr, prs = compute_map(ranks, labels, kappas)

# Map nhãn dựa trên số lượng ảnh (0: Normal (885), 1: Pneumonia (594), 2: Covid (65))
class_names = {0.0: 'Normal', 1.0: 'Pneumonia', 2.0: 'Covid'}

print("\nBẢNG KẾT QUẢ RETRIEVAL (TRUY VẤN) TRÊN COVIDx ĐÃ SỬA LỖI\n")
print(f"| {'Nhãn':<12} | {'Số lượng':<10} | {'mAP (↑)':<10} | {'P@1 (↑)':<10} | {'P@5 (↑)':<10} |")
print("|" + "-"*14 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|")

for class_idx in unique_labels:
    idx = np.where(labels == class_idx)[0]
    if len(idx) == 0:
        continue
        
    class_mAP = np.nanmean(aps[idx]) * 100.0
    class_p1 = np.nanmean(prs[idx, 0]) * 100.0
    class_p5 = np.nanmean(prs[idx, 1]) * 100.0
    count = len(idx)
    
    # Hiển thị tên nhãn cho đẹp báo cáo
    name = class_names.get(class_idx, str(class_idx))
    
    print(f"| {name:<12} | {count:<10} | {class_mAP:<10.2f} | {class_p1:<10.2f} | {class_p5:<10.2f} |")

print(f"\n[*] Trung bình toàn tập (Global mAP): {global_mAP * 100.0:.2f}%")