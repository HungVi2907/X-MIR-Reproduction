# X-MIR Reproduction — Project File Organization

## 📋 Project Structure by Category

---

## 1. 🧠 **Core Model & Architecture**

**Mục đích:** Định nghĩa mô hình deep learning, loss function, và thành phần chính của X-MIR.

| File | Mục đích | Ý nghĩa |
|------|---------|--------|
| `model.py` | Định nghĩa DenseNet-121 embedding model | Backbone chính: embedding-based medical image retrieval |
| `loss.py` | Loss function (triplet loss, etc.) | Huấn luyện metric learning: tối ưu khoảng cách feature |
| `explanations.py` | Saliency map methods (SBSM, SimAtt, SimCAM) | Giải thích hình ảnh → highlight vùng quan trọng |
| `evaluation.py` | Causal metrics (Insertion/Deletion AUC) | Đánh giá chất lượng saliency maps (quantitative) |
| `gradcam.py` | Gradient-based visualization | Hỗ trợ tính toán gradient activation maps |
| `sampler.py` | Batch sampling (triplet sampling) | Lấy mẫu anchor-positive-negative cho training |

---

## 2. 🏃 **Training & Testing Pipeline**

**Mục đích:** Script chạy quy trình huấn luyện và kiểm tra mô hình.

| File | Mục đích | Ý nghĩa |
|------|---------|--------|
| `train.py` | Huấn luyện embedding model trên dataset | Chạy training loop, save checkpoint |
| `test.py` | Kiểm tra & trích xuất embedding, tính mAP/P@K | Đánh giá retrieval performance |
| `compute_saliency.py` | Tính toán saliency maps cho tập test | Sinh file `.npy` (saliency heatmaps) |
| `evaluate_saliency.py` | Đánh giá insertion/deletion AUC | Chạy causal metrics trên saliency maps |
| `anomaly/test_anomaly.py` | Test anomaly mode (xóa class từ training) | Kiểm tra mô hình khi một class bị ẩn |

---

## 3. 📊 **Data Preparation & Utility**

**Mục đích:** Xử lý dữ liệu đầu vào, lấy metadata, hỗ trợ utility.

| File | Mục đích | Ý nghĩa |
|------|---------|--------|
| `read_data.py` | Load ảnh & metadata từ dataset | Chuẩn hóa đầu vào (resize, normalize) |
| `check_isic_order.py` | Kiểm tra thứ tự ảnh ISIC | Debug: đảm bảo consistency |
| `find_isic_list.py` | Tìm danh sách ảnh ISIC từ CSV | Tách subset test/train |
| `find_query.py` | Tìm ảnh query & retrieved neighbors | Hỗ trợ inspect retrieval results |
| `get_metrics.py` | Tính mAP, P@K từ retrieval results | Utility tính evaluation metrics |
| `print_table.py` | In bảng kết quả formatted | Pretty-print results |
| `utils/device.py` | Device abstraction (GPU/MPS/CPU) | Chọn device phù hợp (portable) |

---

## 4. 📝 **Configuration & Metadata Files**

**Mục đích:** Dữ liệu cấu hình, danh sách split, CSV metadata.

| File | Mục đích | Ý nghĩa |
|------|---------|--------|
| `train_split.txt` | Danh sách ảnh huấn luyện COVID | Định nghĩa training set |
| `test_split.txt` | Danh sách ảnh kiểm tra COVID | Định nghĩa test set |
| `test_COVIDx4.txt` | Danh sách ảnh COVIDx4 test | Phiên bản mới của COVIDx dataset |
| `ISIC-2017_Training_Part3_GroundTruth.csv` | Metadata training ISIC | Label & patient ID cho training |
| `ISIC-2017_Test_v2_Part3_GroundTruth_balanced.csv` | Metadata test ISIC | Label cân bằng cho test |
| `requirement.txt` | Dependency list (PyTorch, etc.) | Cài đặt môi trường |

---

## 5. 📊 **Evaluation Results & Output**

**Mục đích:** Kết quả định lượng, JSON metrics, report.

| File | Mục đích | Ý nghĩa |
|------|---------|--------|
| `inser_dele_covid_simatt.json` | Insertion/Deletion scores COVID-19 | Per-image causal metrics |
| `inser_dele_wacv_test_simatt.json` | Insertion/Deletion scores ISIC-2017 | Per-image causal metrics |
| `key_list_covid_simatt.json` | Mapping: image ID → saliency filename COVID | Trace saliency files |
| `key_list_wacv_test_simatt.json` | Mapping: image ID → saliency filename ISIC | Trace saliency files |
| `table.md` | Bảng kết quả retrieval & XAI | Tóm tắt performance cuối cùng |
| `table.md` (tính toán) | `compute_table1.py`, `compute_table2.py` | Script tính & format bảng |
| `ISIC_RESULTS_REPORT.md` | Report kết quả ISIC | Phân tích khác biệt ISIC vs baseline |
| `COVIDx_RESULTS_REPORT.md` | Report kết quả COVIDx | Phân tích chi tiết retrieval và XAI cho dataset COVIDx |
| `evaluation.md` | Tài liệu evaluation metrics | Giải thích Insertion/Deletion/mAP |

---

## 6. 📁 **Directories (Folders)**

**Mục đích:** Lưu trữ dữ liệu lớn, checkpoint, kết quả.

| Thư mục | Mục đích | Nội dung |
|---------|---------|---------|
| `data/` | Dữ liệu input (ảnh) | ISIC-2017 ảnh test, COVID ảnh test |
| `train/`, `test/` | Dữ liệu huấn luyện / kiểm tra | Ảnh gốc raw |
| `checkpoints/` | Lưu model weights | `.pth` / `.pt` files (mô hình đã huấn luyện) |
| `saliency_results/` | Output saliency maps | `.npy` files (heatmaps) |
| `results/` | Kết quả retrieval / metrics | `.npz` hoặc `.json` files |
| `notebooks/` | Jupyter notebooks | Exploratory analysis, visualization |
| `visualize/` | Script visualize | Heatmap overlay, neighbor galleries |
| `anomaly/` | Anomaly detection variant | Riêng biệt: model training khi xóa class |
| `utils/` | Utility modules | Helper functions (device, etc.) |

---

## 7. 🎨 **Documentation & Visualization**

**Mục đích:** Giải thích, hướng dẫn, tài liệu.

| File | Mục đích | Ý nghĩa |
|------|---------|--------|
| `README.md` | Hướng dẫn chính của project | Setup, training, testing steps |
| `LICENSE` | Giấy phép | Usage terms (Air Force DARPA) |
| `xmir_visualizer.py` | Visualize saliency overlays | Heatmap on original image |
| `TargetofEfiles.md` | Bạn đang đọc này | Tóm tắt mục đích từng file/folder |

---

## 8. 📚 **Notebook & Development**

**Mục đích:** Exploratory analysis, preprocessing, debugging.

| File | Mục đích | Ý nghĩa |
|------|---------|--------|
| `create_COVIDx.ipynb` | Load & preprocess COVIDx dataset | Chuẩn bị dữ liệu train/test split |
| `process_ricord.ipynb` | Process RICORD dataset | Exploratory: thử nghiệm dataset mới |

---

## 9. 🔧 **Build/Git**

**Mục đích:** Version control, build config.

| File | Mục đích | Ý nghĩa |
|------|---------|--------|
| `.git/` | Git repository | Version history |
| `.gitignore` | Exclude files | Bỏ qua cache, weights lớn |

---

## 📊 **Typical Workflow**

```
1. [Data Prep]  create_COVIDx.ipynb → train_split.txt, test_split.txt
                ↓
2. [Training]   train.py  (reads: train_split.txt, model.py, loss.py, sampler.py)
                → checkpoints/ (save .pth)
                ↓
3. [Testing]    test.py  (reads: checkpoints/, test_split.txt)
                → results/ (mAP, P@K)
                ↓
4. [Saliency]   compute_saliency.py  (reads: results/, explanations.py)
                → saliency_results/ (.npy maps)
                ↓
5. [Evaluate]   evaluate_saliency.py  (reads: saliency_results/, evaluation.py)
                → inser_dele_*.json, key_list_*.json
                ↓
6. [Report]     compute_table*.py  (reads: JSON results)
                → table.md, SALIENCY_RESULTS_REPORT.md
```

---

## 🎯 **Quick File Lookup**

| Tôi cần... | Tìm ở... |
|-----------|---------|
| Định nghĩa mô hình | `model.py`, `explanations.py` |
| Huấn luyện | `train.py`, `loss.py`, `sampler.py` |
| Kiểm tra retrieval | `test.py`, `get_metrics.py` |
| Tính saliency | `compute_saliency.py`, `explanations.py` |
| Đánh giá saliency | `evaluate_saliency.py`, `evaluation.py` |
| Xem kết quả | `table.md`, `*.md` reports |
| Chuẩn bị dữ liệu | `create_COVIDx.ipynb`, `read_data.py` |
| Chọn device (GPU) | `utils/device.py` |

---

**Tổng kết:** Project X-MIR gồm ~40 file chính, tổ chức thành 9 category từ core model → evaluation → report. Mỗi category có mục đích rõ ràng và liên kết chặt chẽ theo workflow trên.
