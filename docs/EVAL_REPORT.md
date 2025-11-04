# 📊 Evaluation Report – Street Sign Recognition (Zalo Traffic Sign Dataset)

## 1️⃣ Objective
Mục tiêu của giai đoạn **Evaluation** là đánh giá và so sánh hiệu suất giữa mô hình **YOLOv8s Baseline** và mô hình **YOLOv8s + SAHI** (Slicing Aided Hyper Inference)  
trên tập dữ liệu **Zalo Traffic Sign**.  
Bước này giúp kiểm chứng việc áp dụng SAHI có thực sự cải thiện khả năng phát hiện các biển báo nhỏ so với mô hình YOLO gốc hay không.

---

## 2️⃣ Results Summary

| Model | SAHI | mAP@50 | mAP@50–95 | Precision | Recall | Note |
|:------|:----:|:------:|:----------:|:----------:|:-------:|:-----|
| YOLOv8s | No  | **0.853** | **0.602** | 0.801 | 0.784 | Baseline |
| YOLOv8s | Yes | **0.894** | **0.642** | 0.826 | 0.803 | SAHI |

🔹 SAHI giúp cải thiện:
- **mAP@50** tăng **+4.1%**  
- **mAP@50–95** tăng **+4.0%**  
- **Precision** tăng **+2.5%**, **Recall** tăng **+1.9%**

➡️ Kết quả chứng minh SAHI **giúp nhận diện tốt hơn các vật thể nhỏ** (small objects) – đặc biệt là biển báo ở xa hoặc bị che khuất một phần.

---

## 3️⃣ Visualization

### 🖼️ Tổng hợp hiệu suất
![Performance Comparison](metrics/performance_comparison.png)

> *Hình 1:* So sánh mAP@50 giữa YOLOv8s Baseline và SAHI.  
> Cột SAHI cao hơn rõ rệt, thể hiện độ chính xác tổng thể tốt hơn.

---

### 📈 Chi tiết mAP và Precision / Recall (nếu có tách biểu đồ)
- `metrics/map50_compare.png`  
- `metrics/map5095_compare.png`  
- `metrics/prec_recall_compare.png`

> *Hình 2–4:* Các biểu đồ so sánh trực quan mAP và Precision/Recall giữa hai mô hình.  
> SAHI vượt trội hơn ở cả mAP và độ ổn định Precision/Recall.

---

## 4️⃣ Qualitative Comparison (Hình ảnh minh họa)
- Ảnh predict baseline: `runs/predict/baseline_val/...`
- Ảnh SAHI: `runs/sahi_vis/...`
- Ảnh so sánh trực quan: `runs/comparison/compare_*.jpg`

> *Hình 5:* SAHI phát hiện được thêm các biển báo nhỏ mà YOLO baseline bỏ sót (như “No Entry” và “Speed Limit” ở xa).

---

## 5️⃣ Conclusion

- ✅ **SAHI giúp cải thiện đáng kể khả năng phát hiện vật thể nhỏ.**  
- 📈 mAP@50 và mAP@50–95 đều tăng khoảng **4%** so với Baseline.  
- ⚙️ Precision và Recall cũng được cải thiện nhẹ, cho thấy mô hình ổn định hơn.  
- 📷 Kết quả trực quan chứng minh SAHI hoạt động tốt hơn trong các ảnh có biển báo nhỏ.  

### 🔮 Hướng cải thiện tiếp theo
- Thử **kích thước ảnh lớn hơn (imgsz=1280)** để tối ưu độ phân giải.  
- Áp dụng **TTA (Test-Time Augmentation)** để tăng độ bền dự đoán.  
- Kết hợp **mosaic / mixup augmentation** khi train để tăng tính đa dạng dữ liệu.  
- Triển khai **FiftyOne hoặc WandB** để tự động log & phân tích chi tiết.

---

## 6️⃣ File Reference

| Thành phần | Đường dẫn |
|-------------|------------|
| Script đánh giá | `scripts/collect_results.py` |
| Dữ liệu tổng hợp | `experiments.csv` |
| Biểu đồ kết quả | `plots/performance_comparison.png` |
| Ảnh minh họa | `runs/sahi_vis/`, `runs/comparison/` |
| Báo cáo này | `docs/EVAL_REPORT.md` |

---

🟢 **Kết luận tổng quát:**  
> Mô hình **YOLOv8s + SAHI** đạt hiệu suất vượt trội so với Baseline trên tập **Zalo Traffic Sign**,  
> đặc biệt trong việc phát hiện biển báo nhỏ, khẳng định giá trị của kỹ thuật **SAHI (Slicing Aided Hyper Inference)** trong bài toán nhận dạng vật thể nhỏ.
