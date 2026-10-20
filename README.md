# nckh_2 - Pha 1: tai tao 3D 5 mat chiec hop

Windows - Python 3.10.11 - Poetry - RTX 4050 6GB.

## Cai dat
```powershell
poetry env use 3.10
poetry install
```

## Chay
```x64 native
cd /d D:\nckh_2                                  
.\.venv\Scripts\activate                         # Set moi truong

poetry run nckh                                  # menu: [1] quay truc tiep  [2] dung mau co san
poetry run nckh --source record --name box_01    # quay: R bat dau/dung, Q thoat
poetry run nckh --source sample --name box_01    # dung mau da luu
poetry run nckh --source sample --name mouse_01 --prompt mouse  # vat the cong/phi phang voi prompt tuy chon
poetry run nckh --source sample --name mouse_01 --start 5 --until 5 # chay rieng buoc 5 lam min sieu toc (1-2s)
```
Dung mau co san: dat video vao `data/samples/<ten_mau>/video.mp4`.
Vat the phi phang / be mat cong: co the chi dinh `--prompt "<tu_khoa>"` (vi du: `--prompt mouse`, `--prompt cup`),
he thong se tu dong nhan dien vat the, mo rong padding bien cong, lam muot point cloud MLS, dung mesh Poisson va tinh chinh lam min mesh chuyen sau.
Quay bang dien thoai: dung app IP Webcam / DroidCam roi doi `record.camera` trong `configs/default.yaml`
thanh URL (hoac quay san roi dung che do mau).

## Cau truc
- `data/samples/<mau>/` : video.mp4, frames_sharp/, masks/, colmap/ (moi mau 1 thu muc)
- `outputs/<mau>/`      : .ply, .obj, _raw.obj (goc), _refined.obj (da tinh chinh)
- `models/`             : moi loai model 1 thu muc con (grounding_dino, sam2, nafnet); cache HF/torch cung nam day
- `configs/default.yaml`: tham so
- `src/nckh_2/`         : step1..step5, input_source.py (quay / chon mau), main.py

## Tien do
- [x] Buoc 1: loc frame mo
- [x] Buoc 2: Grounded-SAM 2 (nhan dien vat the da prompt + padding bien cong)
- [x] Buoc 3: COLMAP + splatfacto / colmap_mvs
- [x] Buoc 4: Tien xu ly MLS point cloud + Open3D Poisson Reconstruction
- [x] Buoc 5: Pipeline tinh chinh & lam min mesh chuyen sau (Taubin + Polish + Subdivide)
