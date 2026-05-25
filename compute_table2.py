import json
import numpy as np
from pathlib import Path

def calculate_auc_from_json(json_path):
    """
    Tính AUC cho insertion/deletion từ file JSON.
    
    Cấu trúc dự kiến (COVID):
    {
        "image_id_1": [[[ins_scores], [del_scores]], ...],
        "image_id_2": [[[ins_scores], [del_scores]], ...],
    }
    
    Cấu trúc dự kiến (ISIC):
    {
        "image_id_1": [[ins_scores], [del_scores]],
        "image_id_2": [[ins_scores], [del_scores]],
    }
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_insertions = []
    all_deletions = []
    
    # Duyệt qua từng ảnh trong dataset
    for image_id, value in data.items():
        # Xác định cấu trúc: COVID có list of evals, ISIC có single eval
        evaluations = value if isinstance(value[0], list) and isinstance(value[0][0], list) else [value]
        
        # Duyệt qua từng evaluation entry
        for eval_entry in evaluations:
            # eval_entry có thể là [[ins], [del]] hoặc [ins, del]
            if len(eval_entry) >= 2:
                ins_list = eval_entry[0]  # Danh sách insertion scores
                del_list = eval_entry[1]  # Danh sách deletion scores
                
                # Mỗi phần tử có thể là [score] hoặc score
                if isinstance(ins_list, list):
                    all_insertions.extend(ins_list)
                else:
                    all_insertions.append(ins_list)
                    
                if isinstance(del_list, list):
                    all_deletions.extend(del_list)
                else:
                    all_deletions.append(del_list)
    
    if len(all_insertions) == 0 or len(all_deletions) == 0:
        raise ValueError(f"Không đủ dữ liệu để tính AUC từ file: {json_path}")
    
    ins_arr = np.array(all_insertions)
    del_arr = np.array(all_deletions)
    
    # Tính AUC (diện tích dưới đường cong)
    # Sắp xếp và tính trung bình trực tiếp (đây là cách đơn giản để ước tính AUC)
    ins_auc = np.mean(ins_arr)
    del_auc = np.mean(del_arr)
    
    return ins_auc * 100, del_auc * 100

# =====================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# =====================================================================
covid_json = r'D:\tranh\Documents\Study\DataMining\X-MIR-Reproduction\inser_dele_covid_simatt.json'
isic_json = r'D:\tranh\Documents\Study\DataMining\X-MIR-Reproduction\inser_dele_wacv_test_simatt.json'

print("=====================================================================")
print("                 BẢNG 2: QUANTITATIVE EVALUATION (XAI)               ")
print("=====================================================================")
print(f"{'Dataset':<15} | {'Insertion (AUC) ↑':<18} | {'Deletion (AUC) ↓':<18}")
print("-" * 55)

# Tính cho COVID
try:
    c_ins, c_del = calculate_auc_from_json(covid_json)
    print(f"{'COVID-19':<15} | {c_ins:<18.2f} | {c_del:<18.2f}")
except Exception as e:
    print(f"{'COVID-19':<15} | Lỗi: {e}")

# Tính cho ISIC
try:
    i_ins, i_del = calculate_auc_from_json(isic_json)
    print(f"{'ISIC 2017':<15} | {i_ins:<18.2f} | {i_del:<18.2f}")
except Exception as e:
    print(f"{'ISIC 2017':<15} | Lỗi: {e}")
print("=====================================================================")