# X-MIR Reproduction Results

## Bảng 1: Retrieval Results

| Dataset | Model | mAP ↑ | P@1 ↑ | P@5 ↑ |
|---------|-------|-------|-------|-------|
| COVID-19 | DenseNet-121 | 86.9 | 88.4 | 89.0 |
| ISIC 2017 | DenseNet-121 | 54.2 | 62.6 | 62.3 |

## Bảng 2: Quantitative Evaluation (XAI)

| Dataset | Insertion (AUC) ↑ | Deletion (AUC) ↓ |
|---------|-------------------|------------------|
| COVID-19 | 81.62 | 77.38 |
| ISIC 2017 | 76.83 | 62.25 |

---

### Nhận xét

**Bảng 1 - Retrieval Performance:**
- COVID-19 model hoạt động rất tốt (mAP 86.9%, P@1 88.4%).
- ISIC 2017 model thấp hơn (mAP 54.2%, P@1 62.6%), điều này phù hợp với tổn thương da phức tạp hơn so với phân loại X-ray.

**Bảng 2 - Explainability (Saliency Maps):**
- **COVID-19:** Saliency maps có signal mạnh (insertion 81.62%, deletion 77.38%) → model dựa vào các vùng được highlight.
- **ISIC 2017:** Insertion tương đương (76.83%), nhưng deletion thấp hơn (62.25%) → model phụ thuộc vào nhiều feature, không chỉ vùng highlight.
