import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def process_xmir_image(img_path, npy_path, border_color_rgb):
    if not Path(img_path).exists():
        return np.zeros((224, 224, 3), dtype=np.uint8)
        
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    img_array = np.array(transform(img))

    if npy_path and Path(npy_path).exists():
        saliency = np.load(npy_path)
        saliency = saliency - np.min(saliency)
        saliency = saliency / (np.max(saliency) + 1e-8)
        
        saliency_uint8 = np.uint8(255 * saliency)
        heatmap = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        final_img = cv2.addWeighted(img_array, 0.5, heatmap, 0.5, 0)
    else:
        final_img = img_array 

    border_size = 10
    img_with_border = cv2.copyMakeBorder(
        final_img, 
        top=border_size, bottom=border_size, left=border_size, right=border_size, 
        borderType=cv2.BORDER_CONSTANT, 
        value=border_color_rgb
    )
    return img_with_border

# =====================================================================
# CONFIG ĐƯỜNG DẪN THƯ MỤC ISIC
# =====================================================================
isic_path = r'D:\tranh\Documents\Study\DataMining\X-MIR-Reproduction\data\ISIC2017\ISIC-2017_Test_v2_Data'
base_img_dir = Path(isic_path)
base_sal_dir = Path(isic_path)
npz_path = r'D:\tranh\Documents\Study\DataMining\X-MIR-Reproduction\results\isic_densenet121_embed_256_seed_0_epoch_20_ckpt.npz'

# File danh sách 270 ảnh mà bạn vừa tìm được
isic_csv_list = r'D:\tranh\Documents\Study\DataMining\X-MIR-Reproduction\ISIC-2017_Test_v2_Part3_GroundTruth_balanced.csv'

# Màu sắc và Label mapping chuẩn ISIC
BLUE_QUERY  = (0, 112, 192)
GREEN_RIGHT = (0, 176, 80)
RED_WRONG   = (255, 0, 0)
class_mapping = {0: 'Nevus', 1: 'Keratosis', 2: 'Melanoma'}

# =====================================================================
# TỰ ĐỘNG PHÂN TÍCH VÀ TRÍCH XUẤT 
# =====================================================================
print("--> Đang quét radar tìm file .npy trong thư mục...")
all_npy_files = list(base_sal_dir.rglob('*.npy'))
npy_path_map = {}
for p in all_npy_files:
    ten_anh_goc = p.name.replace('.npy', '')
    npy_path_map[ten_anh_goc] = str(p)
print(f"--> Đã lập bản đồ đường dẫn cho {len(npy_path_map)} file Heatmap.")

print("--> Đang đọc cấu trúc dữ liệu xếp hạng từ CSV...")
image_names = []
labels_text = []

# Logic đọc file CSV của ISIC
with open(isic_csv_list, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    # Bỏ qua dòng header đầu tiên (image_id,melanoma,seborrheic_keratosis...)
    for line in lines[1:]: 
        parts = line.strip().split(',')
        if len(parts) >= 3:
            # Cột đầu tiên (parts[0]) là tên ảnh (VD: ISIC_0012086)
            # Vì trong thư mục đuôi file có thể là .jpg, ta thêm cứng đuôi .jpg vào
            filename = parts[0] + '.jpg'
            image_names.append(filename)

data = np.load(npz_path)
dists = data['dists']
labels = np.array(data['labels']).flatten()
ranks = np.argsort(dists, axis=0)

def check_query_npy_exists(q_idx):
    return image_names[q_idx] in npy_path_map

print("--> Đang tìm ca Success/Failure...")
success_q_idx = None
for q in range(len(labels)):
    top_3 = ranks[1:4, q]
    if np.all(labels[top_3] == labels[q]):
        if check_query_npy_exists(q):
            success_q_idx = q
            break

failure_q_idx = None
# Tìm ca thất bại (VD: Melanoma (nhãn 2) nhưng tìm nhầm sang bệnh khác)
for q in range(len(labels)):
    top_3 = ranks[1:4, q]
    if labels[q] == 2.0 and np.all(labels[top_3] != labels[q]):
        if check_query_npy_exists(q):
            failure_q_idx = q
            break

# Nếu không tìm thấy ca sai hoàn toàn cả 3 top, hạ tiêu chuẩn xuống: chỉ cần 1 trong 3 ca sai
if failure_q_idx is None:
    for q in range(len(labels)):
         top_3 = ranks[1:4, q]
         # Có ít nhất 1 ảnh trong Top 3 khác nhãn Query
         if np.any(labels[top_3] != labels[q]): 
             if check_query_npy_exists(q):
                 failure_q_idx = q
                 break

if success_q_idx is None or failure_q_idx is None:
    print("CẢNH BÁO: Dữ liệu Saliency quá ít, không tìm đủ ảnh để vẽ.")
    exit()

selected_queries = [
    {"idx": success_q_idx, "type": "A. Example of Correct Retrieval (Success Case)"},
    {"idx": failure_q_idx, "type": "B. Example of Incorrect Retrieval (Failure Case)"}
]

# =====================================================================
# VẼ ẢNH
# =====================================================================
fig, axes = plt.subplots(2, 4, figsize=(16, 9))

for row_idx, q_data in enumerate(selected_queries):
    q = q_data["idx"]
    row_title = q_data["type"]
    
    top_3_indices = ranks[1:4, q]
    all_images_in_row = [q] + list(top_3_indices)
    fig.text(0.05, 0.91 - (row_idx * 0.46), row_title, fontsize=16, fontweight='bold', ha='left')

    for col_idx, img_idx in enumerate(all_images_in_row):
        ax = axes[row_idx, col_idx]
        img_name = image_names[img_idx]
        current_label = labels[img_idx]
        
        if col_idx == 0:
            border_color = BLUE_QUERY
            caption = f"Query\n({class_mapping[current_label]})"
        else:
            if current_label == labels[q]:
                border_color = GREEN_RIGHT
                caption = f"Top-{col_idx} (Correct)\n({class_mapping[current_label]})"
            else:
                border_color = RED_WRONG
                caption = f"Top-{col_idx} (Incorrect)\n({class_mapping[current_label]})"
                
        img_path = base_img_dir / img_name
        npy_path = npy_path_map.get(img_name, None)
        
        processed_img = process_xmir_image(str(img_path), npy_path, border_color)
        
        ax.imshow(processed_img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(caption, fontsize=12, y=-0.28, pad=10)
        
        for spine in ax.spines.values():
            spine.set_visible(False)

plt.subplots_adjust(wspace=0.05, hspace=0.45)
output_filename = 'X-MIR_ISIC_Visualization.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[XONG] Đã tạo thành công bức ảnh toàn diện: {output_filename}")
plt.show()