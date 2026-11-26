# NCKH 2 - 3D Spatial Reconstruction Pipeline

Hệ thống nghiên cứu và kỹ thuật tái tạo không gian 3D chất lượng cao (3D Spatial Reconstruction) từ video RGB / camera di động qua Wi-Fi LAN, hỗ trợ từ vật thể đơn lẻ (Single-Object) đến mở rộng không gian đa vật thể và toàn cảnh (Full-Scene).

---

## 1. Yêu cầu môi trường & Cài đặt

- **Hệ điều hành**: Windows 10 / 11 (64-bit)
- **GPU**: NVIDIA RTX (khuyến nghị VRAM >= 6GB, ví dụ RTX 4050 / 3060 trở lên), CUDA 12.1
- **Python**: 3.10.11
- **Quản lý gói**: [Poetry](https://python-poetry.org/)

```powershell
# Di chuyen vao thu muc du an
cd /d D:\nckh_2

# Khoi tao moi truong ao Python 3.10
poetry env use 3.10

# Cai dat toan bo thu vien
poetry install
```

---

## 2. Hướng dẫn sử dụng CLI

Kích hoạt môi trường hoặc chạy trực tiếp qua `poetry run`:

```powershell
# Kich hoat moi truong ao
.\.venv\Scripts\activate

# Menu tuong tac (chon quay truc tiep, camera dien thoai hoac dung mau co san)
poetry run nckh
```

### Các lệnh phổ biến (1 CLI duy nhất `poetry run nckh`)

```powershell
# 1. Pipeline Video -> 3D tron bo 5 buoc (da gom Remesh 6 mat + Kiem dinh JSON/PNG)
poetry run nckh --source sample --name mouse_02 --prompt "pink computer mouse" --start 1 --until 5

# 2. Chay lai rieng Buoc 5 voi ho so cau hinh rieng (curved_object | sharp_cad | scene_multi)
poetry run nckh --source sample --name mouse_02 --start 5 --until 5 --profile curved_object
poetry run nckh --source sample --name box_01   --start 5 --until 5 --profile sharp_cad

# 3. Quay video moi (Webcam hoac Camera dien thoai qua QR/Wi-Fi LAN)
poetry run nckh --source record --name box_01
poetry run nckh --source mobile --name box_01 --only-record

# 4. Tinh chinh 1 file .obj / .ply roi bat ky (khong can video/COLMAP)
poetry run nckh --input path\to\model.obj --output path\to\out_dir --reference path\to\ref.obj

# 5. Chi chan doan lech truc PCA/OBB & mat do 6 mat (khong sua file)
poetry run nckh --input path\to\model.obj --output path\to\out_dir --audit-only

# 6. Chay hang loat (Batch) tren ca thu muc nhieu vat the
poetry run nckh --batch-dir path\to\models_dir --output path\to\batch_out --pattern "**\*.obj" --profile curved_object
```

---

## 3. Các bước trong Pipeline

Hệ thống được thiết kế theo dạng đường ống 5 bước độc lập, có thể chạy nối tiếp hoặc tách rời:

| Bước | Module | Chức năng chính | Đầu vào | Đầu ra |
| :--- | :--- | :--- | :--- | :--- |
| **Bước 1** | `pipeline/blur_filter.py` | Lọc frame mờ đa chỉ tiêu (Tenengrad, Isotropy, Laplacian), phạt rung giật (anti-shake), lấy mẫu theo quang đường di chuyển và khử nhiễu ISO giữ cạnh (Bilateral filter). | `video.mp4` | `frames_sharp/*.jpg`, `blur_scores.csv` |
| **Bước 2** | `pipeline/masking.py` | Nhận diện đối tượng theo text prompt bằng Grounding DINO kết hợp SAM 2 video predictor lan truyền 2 chiều; tự động lấp lỗ hổng tem nhãn và bảo toàn quai xách/chi tiết mảnh. | `frames_sharp/*.jpg` | `masks/*.png`, `cutouts/*.jpg` |
| **Bước 3** | `pipeline/reconstruct.py` | Khởi tạo pose camera bằng COLMAP SfM (hỗ trợ cả sequential và exhaustive matcher), phát hiện cảnh báo Pose Collapse / sóng biển, tái tạo dense point cloud bằng COLMAP MVS hoặc Nerfstudio splatfacto (3DGS). | `frames_sharp/`, `masks/` | `point_cloud.ply`, `nerfstudio/` |
| **Bước 4** | `pipeline/mesh_export.py` | Lọc nhiễu point cloud (Statistical + Radius + DBSCAN), nhận diện mặt phẳng bàn (RANSAC), căn thẳng trục tọa độ (Z-up, Z=0), dựng mesh bằng Ball-Pivoting (BPA) và Screened Poisson, đóng kín đáy watertight, tự động xuất 4 hình kiểm định thông số kỹ thuật. | `point_cloud.ply` | `<name>.obj`, `point_cloud_clean.ply`, hình thông số 2-5 |
| **Bước 5** | `pipeline/mesh_refine.py` | Tinh chỉnh & làm mịn bề mặt đa tầng bảo toàn cạnh góc sắc nét (Bilateral Normal Filtering / PyMeshLab Two-Step / Taubin), chuyển giao micro-detail từ điểm quan sát thực và bảo vệ tem nhãn/chi tiết kiến trúc. | `<name>_raw.obj` | `<name>_refined.obj`, `<name>.obj` |
| **Mở rộng** | `pipeline/completion/` | Hoàn thiện vùng che khuất (Retrieval-Augmented Depth-Free 3D Completion): Visual Hull carving từ mask, tìm kiếm & xác thực ảnh web tương đồng (3D Visual RAG), phép chiếu Biharmonic depth proxy và hợp nhất điểm 3D có trọng số tin cậy. | `cutouts/`, `masks/` | `point_cloud_completed.ply`, `prior.json` |

---

## 4. Cấu trúc mã nguồn (Modular Architecture)

Dự án được tái cấu trúc theo mô hình phân tầng hướng miền (Domain-Driven Design):

```text
nckh_2/
├── configs/
│   └── default.yaml             # Tham so cau hinh toan he thong
├── data/
│   ├── raw/certs/               # Chung chi SSL tu ky cho camera dien thoai
│   └── samples/<ten_mau>/       # video.mp4, meta.json, frames_sharp/, masks/, cutouts/
├── models/                      # Binary & weights (colmap/, ffmpeg/, sam2/, hf_cache/)
├── outputs/<ten_mau>/           # .obj, .ply, bieu do thong so ky thuat
├── tests/                       # Bo kiem thu tu dong (Unit & Regression Tests)
│   ├── test_imports.py          # Kiem tra toan ven import cac subpackages
│   ├── test_paths_and_config.py # Kiem tra doc lap CWD & phan giai duong dan
│   └── test_geometry_and_metrics.py # Kiem tra hinh hoc Open3D, lam min & metrics
└── src/nckh_2/
    ├── __init__.py              # Export phien ban va cac API chinh
    ├── main.py                  # CLI entrypoint (poetry run nckh)
    ├── core/                    # Nen tang & hop dong du lieu
    │   ├── paths.py             # Dinh vi ROOT tu dong qua pyproject.toml
    │   ├── config.py            # Trinh nap YAML UTF-8
    │   ├── types.py             # SceneContext, SampleMetadata, StageResult
    │   └── interfaces.py        # PipelineStage, Masker, Reconstructor protocols
    ├── capture/                 # Giao tiep thu nhan video & camera
    │   ├── camera_io.py         # Threaded LiveCameraReader & AsyncVideoWriter khong tre
    │   ├── sample_store.py      # Quan ly, liet ke va chon tap mau
    │   ├── recorder.py          # Vong lap OpenCV HUD tuong tac quay video
    │   ├── mobile_camera.py     # Server HTTPS/WebSocket stream tu trinh duyet dien thoai
    │   └── templates/
    │       └── mobile_camera.html # Giao dien Web Camera di dong (HTML5 + getUserMedia)
    ├── backends/                # Tien ich he thong & thuat toan nen
    │   ├── device.py            # Quan ly thiet bi CPU / CUDA GPU & giai phong VRAM
    │   ├── colmap_io.py         # Doc du lieu binary cameras.bin, images.bin, transforms.json
    │   ├── colmap_patch.py      # Patch tuong thich nerfstudio cho COLMAP 4.2.0
    │   ├── geometry.py          # Don sach topo, can truc, dong kin mesh, xuat OBJ
    │   └── smoothing.py         # Bilateral Normal Filtering & Two-step mesh polish
    ├── pipeline/                # Cac giai doan xu ly chinh
    │   ├── runner.py            # Bo dieu phoi pipeline, dang ky cac buoc 1-5
    │   ├── blur_filter.py       # Buoc 1: Loc frame mo & anti-shake
    │   ├── masking.py           # Buoc 2: Grounded-SAM 2
    │   ├── reconstruct.py       # Buoc 3: COLMAP SfM & MVS / splatfacto
    │   ├── mesh_export.py       # Buoc 4: Dung mesh Poisson / BPA
    │   ├── mesh_refine.py       # Buoc 5: Tinh chinh & lam min chuyen sau
    │   └── completion/          # Retrieval-augmented depth-free 3D completion
    └── evaluation/              # Danh gia & kiem dinh ky thuat
        ├── spec_figures.py      # Xuat 4 hinh thong so truc giao & kiem dinh 3D
        └── metrics.py           # SurfaceMetrics (Chamfer, F-score, NC, CPR)
```

---

## 5. Kiểm thử tự động (Verification & Testing)

Dự án tích hợp bộ kiểm thử tự động toàn diện:

```powershell
# Chay toan bo test suite
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Kiem tra lenh tro giup CLI
poetry run nckh --help

# Chay thu nghiem nhanh Buoc 5 tren mau co san
poetry run nckh --source sample --name box_01 --start 5 --until 5
```
