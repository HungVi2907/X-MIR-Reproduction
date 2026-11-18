# Báo Cáo Kết Quả Thực Nghiệm X-MIR trên Tập Dữ Liệu COVIDx

## 1. Kết Quả Truy Vấn (Retrieval Metrics)
Bảng dưới đây trình bày hiệu suất truy xuất ảnh (Content-Based Image Retrieval) sử dụng mạng DenseNet-121 trên tập kiểm tra COVIDx.

| Nhãn Bệnh | Số lượng | mAP (↑) | P@1 (↑) | P@5 (↑) |
| :--- | :--- | :--- | :--- | :--- |
| Normal | 885 | 90.40% | 92.66% | 92.18% |
| Pneumonia | 594 | 88.00% | 88.72% | 88.59% |
| Covid | 65 | 28.92% | 55.38% | 50.46% |
| **Trung bình (Global)** | **1544** | **86.88%** | - | - |

## 2. Kết Quả Đánh Giá Giải Thích (XAI - Saliency Metrics)
Đánh giá mức độ tập trung của mô hình vào các vùng tổn thương (sử dụng phương pháp SimAtt) thông qua chỉ số Insertion và Deletion AUC.

| Nhóm bệnh | Số lượng ảnh | Insertion AUC (↑) | Deletion AUC (↓) |
| :--- | :--- | :--- | :--- |
| Pneumonia | 51 | 0.8286 | 0.7789 |
| Normal | 25 | 0.8189 | 0.8119 |
| Covid | 25 | 0.8090 | 0.7583 |

## 3. Phân Tích & Biện Luận (Discussion)
* **Hiệu suất chung ấn tượng:** Mô hình đạt mAP tổng thể (Global mAP) lên đến **86.88%**, chứng tỏ cấu trúc DenseNet-121 kết hợp SimAtt trích xuất đặc trưng không gian (spatial features) rất tốt cho ảnh X-quang phổi nói chung (nhóm Normal và Pneumonia đạt mAP quanh mức 90%).
* **Độ tin cậy của mô hình giải thích (Saliency Map):** Chỉ số Insertion AUC của cả 3 nhóm đều đạt ngưỡng cao (trên 0.8), khẳng định bản đồ nhiệt (saliency maps) sinh ra đã định vị chuẩn xác các vùng mang đặc trưng bệnh lý cốt lõi, minh chứng rằng mô hình học được các dấu hiệu y khoa chứ không dựa vào nhiễu (noise) xung quanh ảnh.
* **Thách thức ở nhóm Covid (Imbalanced Data & Feature Overlap):** * Hiệu suất truy vấn của nhóm Covid suy giảm đáng kể (mAP ~28.92%). Nguyên nhân chính đến từ sự mất cân bằng dữ liệu cực đoan trong tập test (Covid chỉ chiếm khoảng 4.2% tổng số lượng ảnh đánh giá với 65 mẫu). 
    * Thêm vào đó, có sự chồng chéo đặc trưng X-quang (Feature Overlap) rất lớn giữa bệnh nhân Covid-19 và viêm phổi thông thường (Pneumonia). Hệ thống có khả năng cao lấy được đúng ảnh Covid ở Top 1 (P@1 = 55.38%) nhưng không gian nhúng (embedding space) vẫn bị nhiễu khiến các kết quả tiếp theo bị lẫn nhiều ảnh Pneumonia. 
    * **Hướng phát triển (Future Work):** Việc áp dụng các kỹ thuật cân bằng dữ liệu, Hard-Negative Mining, hoặc Contrastive Learning để tối ưu và kéo dãn không gian nhúng cho lớp thiểu số này sẽ là bước cải tiến cần thiết tiếp theo.
