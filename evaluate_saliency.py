import os
import math
import json
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from model import DenseNet121
from torchvision import transforms
from evaluation import CausalMetric, gkern
from utils.device import get_device
from pathlib import Path

# NOTE: Edit the dataset_type and path to model_weights here
dataset_type = 'covid'
model_weights = './anomaly/'


class InsDel():
    def __init__(self,
                 model):
        self.model = model
        net_in_size = 224
        klen = 51
        ksig = math.sqrt(50)
        kern = gkern(klen, ksig)
        def blur(x): return nn.functional.conv2d(
            x, kern.to(get_device()), padding=klen//2)
        self.insertion = CausalMetric(
            self.model, 'ins', net_in_size, substrate_fn=blur)
        self.deletion = CausalMetric(
            self.model, 'del', net_in_size, substrate_fn=torch.zeros_like)
 
    def evaluate(self, new_sal, ret_image):
        """
        This function evaluates an image and its saliency map for deletion 
        and insertion.

        Attributes:
            new_sal (numpy.array): The saliency map obtained at iteration k. 
            ret_image (torch.tensor.cuda): Input image that needs to be explained.

        Returns:
        score_del (float): The deletion score between 0-1 range.
        score_ins (float): The insertion score between 0-1 range.
        """
        new_sal = torch.from_numpy(new_sal).float()
        score_del = 0
        zero_cnt_del = 0
        score_del, zero_cnt_ins = self.deletion.single_run(self.q_image.cpu(),
                                                           ret_image.cpu(), new_sal, verbose=0)
        score_ins, zero_cnt_del = self.insertion.single_run(self.q_image.cpu(),
                                                            ret_image.cpu(),  new_sal, verbose=0)
        return score_del, score_ins, zero_cnt_ins, zero_cnt_del

    def load_query(self, query_image):
        self.q_image = query_image

    def forward(self, q_image, ret_dict, sal_dict):
        ins_avg = []
        del_avg = []
        z_ins_list = []
        z_del_list = []

        self.load_query(q_image)
        for i, sal_m in enumerate(sal_dict):
            ret_image = ret_dict[i]
            dele, ins, z_ins, z_del = self.evaluate(sal_m, ret_image)
            z_ins_list.append(z_ins)
            z_del_list.append(z_del)
            ins_avg.append(ins)
            del_avg.append(dele)
        return ins_avg, del_avg, z_ins_list, z_del_list


# Class for average counter
class AverageCounter():
    def __init__(self):
        self.average = 0
        self.running_avg = 0
        self.fina_dict = {}

    def store(self, q_label, metric,  k=20):
        def check_if_in_dict(value1):
            if value1 in self.fina_dict.keys():
                return True
            else:
                return False
        if check_if_in_dict(q_label):
            self.fina_dict[q_label].append(metric)
        else:
            self.fina_dict[q_label] = [metric]

    def read_Average(self):
        avgdict = {}
        for k, v in self.fina_dict.items():
            avgdict[k] = sum(v) / float(len(v))
        return avgdict


def prep_image_(file_n):
    query_image = Image.open(os.path.join(
        query_img_path, file_n)).convert('RGB')
    query_image_tensor = transform(query_image).unsqueeze_(0).to(get_device())
    return query_image_tensor


model = DenseNet121()
model_weights = './checkpoints/isic_densenet121_embed_256_seed_0_epoch_20_ckpt.pth'
model.load_state_dict(torch.load(model_weights, map_location='cpu'), strict=False)
model = model.eval()
model = model.to(get_device())

# Logging counter
ins_avg_c = AverageCounter()
del_avg_c = AverageCounter()


sal_map = {}
ret_map = {}
query_image_list = []
class_labels = {}

if dataset_type == 'covid':
    # Dùng Path để triệt tiêu lỗi gạch chéo ngược (\) trên Windows
    main_path = Path(r'D:\tranh\Documents\Study\DataMining\X-MIR-Reproduction\saliency_results\test')
    query_img_path = Path(r'D:\tranh\Documents\Study\DataMining\X-MIR-Reproduction\test')
    valid_class = ['pneumonia', 'normal']
    
    with open('test_split.txt', 'r') as f:
        for line in f.readlines():
            label = line.split()[2]
            if label not in valid_class:
                label = 'covid'
            # Tách lấy tên file phòng trường hợp trong file txt ghi là 'test/anh.png'
            q_na = line.split()[1].split('/')[-1].split('\\')[-1] 
            class_labels[q_na] = label

# =============== ĐOẠN ĐƯỢC CỨU SỐNG ===============
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# ==================================================

# Đổi tên file xuất ra để an toàn
f = open('./inser_dele_covid_simatt.json', 'w')
f2 = open('./key_list_covid_simatt.json', 'w')

ins_del_q_dict = {}
key_dict = {}
get_insert_dele = InsDel(model)

print(f"\n[BƯỚC 1] Đang quét tìm .npy trong: {main_path}")
npy_files = list(main_path.rglob('*.npy'))
print(f"[BƯỚC 2] Số lượng file .npy tìm thấy: {len(npy_files)}\n")

if len(npy_files) == 0:
    print("!!! CẢNH BÁO ĐỎ: Python không nhìn thấy bất kỳ file .npy nào !!!")
    exit()

for sal_path in npy_files:
    file_n_raw = sal_path.name # Lấy tên (VD: 3d895c52.png.npy)
    file_n = file_n_raw.replace('.npy', '')
    base_name = file_n.split('.')[0]
    
    img_path = query_img_path / file_n
    if not img_path.exists():
        print(f"[-] BỎ QUA: Không tìm thấy ảnh gốc tại {img_path}")
        continue
        
    print(f"Processing: {file_n}")
    query_image_tensor = prep_image_(str(img_path.name)) # Đọc ảnh gốc
    
    # Khởi tạo Dictionary
    if base_name not in sal_map:
        sal_map[base_name] = []
        ret_map[base_name] = []
        key_dict[base_name] = []
        
    sal_map[base_name].append(np.load(str(sal_path)))
    ret_map[base_name].append(query_image_tensor)
    key_dict[base_name].append(file_n_raw)

    # Chạy tính toán
    insertion, deletion, i_in, i_del = get_insert_dele.forward(
        query_image_tensor, 
        ret_map[base_name], 
        sal_map[base_name]
    )

    avg_insert = sum(insertion) / len(insertion)
    avg_del = sum(deletion) / len(deletion)
    
    if base_name not in ins_del_q_dict:
        ins_del_q_dict[base_name] = []
    ins_del_q_dict[base_name].append([insertion, deletion])
    
    # Khớp nhãn và lưu điểm
    if file_n in class_labels:
        ins_avg_c.store(class_labels[file_n], avg_insert)
        del_avg_c.store(class_labels[file_n], avg_del)
    else:
        print(f"  -> [Cảnh báo] Ảnh '{file_n}' không có trong file test_split.txt")
        
    # Dọn RAM
    del sal_map[base_name]
    del ret_map[base_name]

# Lưu JSON
json.dump(ins_del_q_dict, f)
json.dump(key_dict, f2)

print("\n--- KẾT QUẢ ĐÁNH GIÁ (INSERTION) ---")
print(ins_avg_c.read_Average())
print("--- KẾT QUẢ ĐÁNH GIÁ (DELETION) ---")
print(del_avg_c.read_Average())
