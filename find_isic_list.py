import glob
import os

project_dir = r'D:\tranh\Documents\Study\DataMining\X-MIR-Reproduction'
print("=== ĐANG KHỞI ĐỘNG RADAR TRUY TÌM DANH SÁCH 270 ẢNH ISIC ===")

# Quét tất cả các file cấu hình và văn bản
for ext in ['*.txt', '*.csv', '*.json']:
    # Lục tung mọi thư mục con
    for filepath in glob.glob(os.path.join(project_dir, '**', ext), recursive=True):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Kiểm tra xem file có chứa từ khóa của tập ISIC không
                if 'ISIC_' in content:
                    lines = content.split('\n')
                    count = sum(1 for line in lines if 'ISIC_' in line)
                    # Nếu tìm thấy file có chứa khoảng 270 tên ảnh, đó chính là mục tiêu!
                    if count > 0:
                        print(f"-> PHÁT HIỆN ẢNH: {filepath}")
                        print(f"   (Chứa {count} dòng có tên ảnh ISIC)")
        except Exception:
            pass

print("=== QUÉT XONG ===")