**Quy trình hoạt động đầy đủ — EMNIST Recognition**

Mục tiêu: mô tả chi tiết các bước vận hành của chương trình, các thành phần liên quan, cách chạy, cấu hình và hướng xử lý sự cố.

1) Tổng quan kiến trúc
- Entry point: `main.py` (chạy `main()`).
- Thư viện & module chính: `src/preprocessing.py`, `src/load_emnist.py`, `src/classifier.py`.
- Dữ liệu: thư mục `data/` (raw EMNIST và cache `emnist_preprocessed.npz`).
- Kết quả: thư mục `models/` (ví dụ `models/pipeline.pkl`).

2) Luồng dữ liệu (chi tiết)
- Load data: `load_emnist(data_dir='./data')` trả về `X_tr, y_tr, X_te, y_te, labels`.
- Preprocessing:
  - Kiểm tra cache `data/emnist_preprocessed.npz`.
  - Nếu không có, gọi `straighten_image()` cho từng ảnh train/test (chậm), xây cache bằng `np.savez()`.
  - Nếu có, load `X_tr_s, X_te_s` từ cache.
- PCA:
  - Gọi `build_pca_pipeline(X_tr_s, variance=0.95)` để fit PCA trên tập huấn luyện.
  - Transform tập test: `X_te_pca = pca.transform(X_te_s)`.
- Huấn luyện:
  - Tạo `RPCAKNNClassifier(n_trees=.., k=.., metric=.., leaf_size=..)` và gọi `fit(X_tr_pca, y_tr)`.
- Đánh giá:
  - `preds = clf.predict(X_te_pca)`
  - Tính `accuracy_score`, `classification_report`.
- Lưu pipeline:
  - `joblib.dump({'pca': pca, 'clf': clf, 'labels': labels}, 'models/pipeline.pkl')`.

3) Các thành phần chi tiết và phụ thuộc
- `pca`: đối tượng `sklearn.decomposition.PCA` (chỉ reliant vào scikit-learn + numpy). Có thể lưu và tải độc lập.
- `clf`: `src.classifier.RPCAKNNClassifier` (lưu bằng pickle phụ thuộc định nghĩa lớp trong `src/classifier.py`).
- `labels`: danh sách các tên lớp (thuần Python).

4) Hướng dẫn vận hành (commands)
- Dùng môi trường ảo và cài dependencies từ `requirements.txt`.

```bash
python -m pip install -r requirements.txt
python main.py
```

Tùy chọn (nếu muốn skip straighten cache): tạo file `data/emnist_preprocessed.npz` trước khi chạy.

5) Cấu hình & tham số
- Nơi thay đổi tham số: `main.py` (tham số PCA variance, `RPCAKNNClassifier` args).
- Tùy chọn CLI (khuyến nghị): thêm argparse vào `main.py` để cho phép truyền `--n-trees`, `--k`, `--variance`, `--cache`.

6) Lưu & tái dùng pipeline an toàn
- Lưu `pca` và `labels` tách riêng để dùng ở môi trường khác mà không cần `src`:
  - `joblib.dump(pca, 'models/pca.pkl')`
  - `joblib.dump(labels, 'models/labels.pkl')`
- `clf` phụ thuộc class định nghĩa trong `src/classifier.py`. Nếu muốn dùng `clf` ở nơi khác, cần:
  - cung cấp cùng file `src/classifier.py`, hoặc
  - export trọng số/ cấu trúc nội bộ (nếu implement hỗ trợ) vào dạng JSON / numpy arrays.

7) Xử lý sự cố phổ biến
- Lỗi khi load pickle (`No module named 'src'`): đảm bảo repository root có trong `sys.path` hoặc cài package/ module tương ứng; script `scripts/repair_pickle.py` đã thêm repo root vào `sys.path` để khắc phục.
- Lỗi `invalid load key '\t'` thường do file không phải pickle hoặc bị hỏng; kiểm tra header bytes (ví dụ `file.read(64).hex()`) để xác định định dạng/compression.
- Nếu gặp pickle phụ thuộc module nội bộ, tải file ở môi trường có module đó trên `PYTHONPATH`.

8) Quy trình phục hồi model bị hỏng (recommended)
1. Chạy `scripts/repair_pickle.py models/pipeline.pkl` để chẩn đoán và thử `joblib.load`/`pickle.load`/giải nén.
2. Nếu `joblib.load` thành công: dùng script để `joblib.dump()` các phần an toàn (`pca`, `labels`) ra file riêng.
3. Nếu `clf` cần được phổ biến: mở `src/classifier.py` và thêm phương thức `to_dict()`/`from_dict()` để trích xuất trạng thái (các cây, ma trận, v.v.) rồi lưu dưới dạng JSON/npy.

9) Kiểm thử & tái tạo (CI)
- Thêm test nhỏ `tests/test_pipeline_save_load.py` để đảm bảo `joblib.load('models/pipeline.pkl')` chạy trên CI (cài `src` vào path trước). Điều này giúp phát hiện sớm lỗi tương thích.

10) Ghi chú vận hành (ngắn)
- Thời gian preprocess lần đầu có thể ~10 phút hoặc hơn; dùng cache.
- Khi chia sẻ model với người khác, thêm `requirements.txt` với phiên bản `scikit-learn` tương thích và hoặc cung cấp module `src` hoặc xuất `clf` nội dung thuần.

File này: `WORKFLOW_FULL.md` (gốc repo).
