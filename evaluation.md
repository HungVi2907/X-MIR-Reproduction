# Phân tích Hiệu năng Giải thích Mô hình X-MIR (Tập ISIC)

## 1. Cơ sở lý thuyết: Causal Metrics (Đo lường Nhân quả)
Để chứng minh AI chẩn đoán bệnh dựa trên vùng da bị tổn thương chứ không phải do học "vẹt" các chi tiết thừa (như vết mực đánh dấu của bác sĩ, màu sắc ánh sáng chụp), hệ thống áp dụng phương pháp đo lường nhân quả thông qua 2 chỉ số: **Insertion** và **Deletion**.

### A. Chỉ số Insertion (Thêm vào)
* **Cơ chế:** Khởi đầu với một bức ảnh bị làm mờ/nhiễu hoàn toàn (độ tự tin của mô hình = 0). Thuật toán sẽ khôi phục lại các điểm ảnh (pixel) theo thứ tự từ vùng có nhiệt độ cao nhất (Màu Đỏ trên Heatmap) đến vùng thấp nhất (Màu Xanh).
* **Ý nghĩa:** Nếu chỉ cần khôi phục vùng Đỏ mà mô hình đã lập tức đưa ra quyết định bệnh chính xác với độ tự tin cao, điều đó chứng tỏ Heatmap đã khoanh đúng trọng tâm mầm bệnh.
* **Tiêu chuẩn:** Điểm số càng **CAO** càng tốt (Hướng tới 1.0).

### B. Chỉ số Deletion (Xóa đi)
* **Cơ chế:** Khởi đầu với bức ảnh gốc rõ nét (độ tự tin cao nhất). Thuật toán tiến hành xóa dần các pixel cũng theo thứ tự từ vùng Đỏ đến vùng Xanh.
* **Ý nghĩa:** Nếu AI đột ngột "mù" và giảm mạnh độ tự tin ngay khi vùng Đỏ bị xóa đi, chứng tỏ AI thực sự phụ thuộc vào vùng đó để chẩn đoán.
* **Tiêu chuẩn:** Điểm số càng **THẤP** càng tốt (Hướng tới 0.0).

---

## 2. Đối chiếu Kết quả Thực nghiệm với Bài báo gốc (WACV 2022)

### A. Kết quả từ Bài báo gốc
Trong báo cáo của tác giả trên tập ISIC, thuật toán SimCAM cho ra xu hướng đặc trưng:
* Điểm **Insertion** luôn duy trì ở mức cao (thường dao động > 0.65 - 0.75).
* Điểm **Deletion** luôn được kéo xuống mức thấp hơn rõ rệt (dao động khoảng 0.50 - 0.60).
* *Kết luận lõi của bài báo:* Khoảng cách (Gap) giữa Insertion và Deletion càng lớn chứng tỏ phương pháp XAI hoạt động càng hiệu quả.

### B. Kết quả Thực nghiệm của Nhóm
Trích xuất từ kết quả chạy thực tế (file `inser_dele_wacv_test_simatt.json`):
* **ISIC_0014059:** Insertion đạt **0.909**, Deletion giảm còn **0.681**.
* **ISIC_0012358:** Insertion đạt **0.888**, Deletion giảm còn **0.537**.
* **ISIC_0013319:** Insertion đạt **0.837**, Deletion giảm sâu về **0.489**.

### C. Đánh giá tính hợp lệ
Kết quả thu được **hoàn toàn đồng nhất với xu hướng của bài báo khoa học**.
Mô hình tái tạo không những bảo toàn được khoảng cách (Gap) logic: `Insertion >> Deletion`, mà ở một số trường hợp cụ thể, chỉ số Insertion còn chạm ngưỡng rất lý tưởng (~0.85 đến 0.90). Điều này khẳng định hệ thống kỹ thuật và bộ trọng số tiền huấn luyện (pretrained weights) đã được thiết lập và khởi chạy một cách chính xác, không làm suy hao logic gốc của tác giả.