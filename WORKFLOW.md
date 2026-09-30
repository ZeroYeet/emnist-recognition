**Luồng Chương Trình (Workflow) — EMNIST Recognition**

Mô tả ngắn: chương trình nhận dạng ký tự tay (EMNIST) theo các bước chính: tải dữ liệu → tiền xử lý (straighten + cache) → giảm chiều bằng PCA → huấn luyện RPCA Forest + KNN → đánh giá → lưu pipeline.

**Bước chính**
- **Load data**: sử dụng `load_emnist()` trong `src/load_emnist.py`.
- **Preprocessing**: gọi `straighten_image()` trong `src/preprocessing.py`; kết quả được cache vào `data/emnist_preprocessed.npz` để tránh tính toán lại.
- **PCA**: `build_pca_pipeline()` tạo và fit PCA (ví dụ `variance=0.95`), transform dữ liệu train/test.
- **Train**: `RPCAKNNClassifier` trong `src/classifier.py` (tham số: `n_trees`, `k`, `metric`, `leaf_size`).
- **Evaluate**: dùng `sklearn.metrics` để tính `accuracy` và `classification_report`.
- **Save**: lưu pipeline (`pca`, `clf`, `labels`) bằng `joblib.dump()` vào `models/pipeline.pkl`.

**Sơ đồ luồng (Mermaid)**

```mermaid
flowchart LR
  A[Load data]
  B[Preprocess\n(straighten + cache)]
  C[PCA\n(fit + transform)]
  D[Train\nRPCA Forest + KNN]
  E[Evaluate\n(metrics)]
  F[Save\npipeline]

  A --> B
  B --> C
  C --> D
  D --> E
  E --> F
  B -->|cache exists| C
```

**Tệp và điểm vào (entry points)**
- Chạy chính: `main.py` (hàm `main()`) — hiện có sẵn.
- Các hàm chính: `src/preprocessing.straighten_image`, `src/preprocessing.build_pca_pipeline`, `src/classifier.RPCAKNNClassifier`, `src/load_emnist.load_emnist`.

**Lưu ý vận hành & tùy biến**
- Cache: nếu `data/emnist_preprocessed.npz` tồn tại thì bỏ qua bước straighten.
- Tham số PCA và classifier có thể điều chỉnh trong `main.py` trước khi huấn luyện.
- Đề xuất: thêm wrapper CLI để truyền tham số (ví dụ `--n-trees`, `--k`, `--cache`) và mục lục lệnh chạy vào `README.md`.

**Lệnh chạy nhanh**

```bash
python main.py
```

File: `WORKFLOW.md` trong thư mục gốc repository.
