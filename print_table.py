import json

with open('inser_dele_covid_simatt.json', 'r') as f:
    data = json.load(f)

labels = {}
valid_class = ['pneumonia', 'normal']
with open('test_split.txt', 'r') as f:
    for line in f.readlines():
        parts = line.split()
        if len(parts) >= 3:
            file_name = parts[1].split('/')[-1].split('\\')[-1]
            base_name = file_name.split('.')[0]
            label = parts[2]
            labels[base_name] = label if label in valid_class else 'covid'

# Tự động gom nhóm dựa trên dữ liệu thực tế
metrics = {}

for base_name, scores in data.items():
    if base_name in labels:
        label = labels[base_name]
        if label not in metrics:
            metrics[label] = {'ins': [], 'del': []} # Tự động thêm nhãn mới nếu có
        try:
            ins_avg = sum(scores[0][0]) / len(scores[0][0])
            del_avg = sum(scores[0][1]) / len(scores[0][1])
            metrics[label]['ins'].append(ins_avg)
            metrics[label]['del'].append(del_avg)
        except:
            pass

print("\nBẢNG KẾT QUẢ ĐÁNH GIÁ XAI TRÊN COVIDx (SIMATT)\n")
print("| Nhóm bệnh | Số lượng ảnh | Insertion AUC (↑) | Deletion AUC (↓) |")
print("| :--- | :--- | :--- | :--- |")
for label, vals in metrics.items():
    if len(vals['ins']) > 0:
        final_ins = sum(vals['ins']) / len(vals['ins'])
        final_del = sum(vals['del']) / len(vals['del'])
        count = len(vals['ins'])
        print(f"| {label.capitalize():<9} | {count:<12} | {final_ins:<17.4f} | {final_del:<16.4f} |")