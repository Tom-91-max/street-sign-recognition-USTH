# Street Sign Recognition - Zalo Traffic Sign 
## How to run 
1\) pip install -r requirements.txt 
2\) python scripts\train.py 
3\) python scripts\predict.py 
feature/data-B
4\) data set:https://drive.google.com/file/d/1c_t28KxcMOdwGIrF-8vC7CzGlAD9pj7j/view?usp=drive_link


link dataset: https://datasetninja.com/zalo-traffic-sign
 main

Python 3.11.9

1.Sau khi git clone dự án về , chạy lệnh này để lấy hết thư viện về:
pip install -r requirements.txt


2. Nhiệm vụ và file làm việc của từng thành viên
B — Data Lead (Xử lý dữ liệu & EDA)
    data/ (chứa ảnh & nhãn)
    configs/zalo.yaml
    notebooks/EDA_and_Training.ipynb

C — Baseline Engineer (Huấn luyện YOLO cơ bản)
    scripts/train.py
    scripts/predict.py
D — Small-Object Specialist (Cải thiện model & SAHI)
    scripts/sahi_infer.py
    (tuỳ chọn) scripts/train_large.py
E — MLOps & Evaluation (So sánh kết quả, logging)
    scripts/collect_results.py
    experiments.csv
    (tuỳ chọn) wandb log.


