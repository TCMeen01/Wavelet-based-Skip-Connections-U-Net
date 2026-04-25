# Wavelet-based Skip Connections U-Net
Dự án này triển khai một ý tưởng cải tiến nhằm tăng cường độ ổn định biên (boundary stability) của mô hình U-Net trong bài toán Phân vùng Ảnh Y khoa (Medical Image Segmentation). Cụ thể, kiến trúc thay thế các Skip Connection truyền thống bằng các kết nối dựa trên biến đổi Wavelet (Wavelet-based Skip Connections), sử dụng phép biến đổi DTCWT (Dual Tree Complex Wavelet Transform).

Tham khảo bài báo U-Net gốc tại đây: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

## Thành viên nhóm
- Trần Công Minh - 23127007 - HCMUS
- Lê Hồng Ngọc - 23127236 - HCMUS

## Cấu trúc thư mục
```text
Wavelet-based-Skip-Connections-U-Net/
├── DataHandle/          # Module xử lý dữ liệu đầu vào
│   ├── DataLoader.py    # Khởi tạo DataLoader cho quá trình huấn luyện/đánh giá
│   ├── Dataset.py       # Cấu trúc lớp Dataset kế thừa từ PyTorch để load ảnh và mask
│   └── Transforms.py    # Các kỹ thuật tiền xử lý và augmentation (dùng Albumentations)
│
├── Datasets/            # Nơi chứa các tập dữ liệu ảnh y khoa (ISIC-2018, Kvasir-SEG)
│
├── Models/              # Thư mục lưu trữ định nghĩa tải pre-trained và các file trọng số (.pth)
│
├── Unet/                # Chứa định nghĩa kiến trúc mạng
│   ├── Unet.py          # Kiến trúc U-Net cơ bản (Baseline)
│   ├── Unet_parts.py    # Các khối layers cơ sở cấu thành mạng (DoubleConv, Encoder, Decoder...)
│   └── WTSC_Unet.py     # Mạng DTCWTSC-UNet đề xuất (sử dụng Wavelet-based Skip Connections)
│
├── Utils/               # Các hàm số phụ trợ, độ đo và mất mát
│   ├── metrics.py       # Các độ đo đánh giá (IoU, Dice, Hausdorff Distance...)
│   ├── objectives.py    # Các hàm mất mát (Loss functions)
│   └── utils.py         # Các hàm phụ trợ, chủ yếu để trực quan hóa kết quả
│
├── Wavelet/             # Module xử lý thành phần Wavelet
│   ├── DTCWT.py         # Định nghĩa cấu hình Dual Tree Complex Wavelet Transform
│   └── test_wavelet.py  # Viết kiểm thử hiển thị thành phần phân rã Wavelet
│
├── inference.py         # Script chạy kiểm thử (inference) và đánh giá mô hình
├── train.py             # Script huấn luyện (training) mô hình
├── requirements.txt     # Danh sách package môi trường cần thiết
└── README.md            # Tài liệu giới thiệu và hướng dẫn dự án
```

## Cài đặt

Yêu cầu bắt buộc là sử dụng phiên bản **`numpy < 2`** để đảm bảo tương thích các hàm tính toán bên dưới, các framework khác được liệt kê chi tiết trong file requirements.

Chạy lệnh sau để cài đặt toàn bộ tự động:
```bash
pip install -r requirements.txt
```

## Kiến trúc và Dữ liệu hỗ trợ

- **Các mô hình có sẵn:** 
  - `Unet` (Baseline U-Net cơ bản)
  - `DTCWTSC_UNet` (Kiến trúc đề xuất tích hợp Wavelet-based Skip Connections)
- **Tập dữ liệu hỗ trợ sẵn:** Mã nguồn hỗ trợ pipeline mạnh mẽ cho 2 tập dữ liệu y y tế phổ biến: `ISIC` (segmentation tổn thương da) và `Kvasir` (polyp nội soi).

## Hướng dẫn chạy

### 1. Cách Huấn luyện (Training)

Sử dụng script `train.py` với các tham số tương ứng qua cửa sổ dòng lệnh.

**Lệnh mẫu:**
```bash
cd Wavelet-based-Skip-Connections-U-Net

python -m train --model DTCWTSC_UNet --dataset_name ISIC --dataset_path Datasets/Kvasir-SEG --n_epochs 50 --criterion BCEDiceLoss --device gpu --Wavelet_Level 1 --model_save_path Models/my_model.pth
```

**Giải thích chi tiết các tham số:**
- `--model` *(bắt buộc)*: Kiến trúc mạng sử dụng để train. Truyền `Unet` hoặc `DTCWTSC_UNet`.
- `--dataset_name` *(bắt buộc)*: Tên tập dữ liệu đang dùng (VD: `ISIC` hoặc `Kvasir`).
- `--dataset_path` *(bắt buộc)*: Đường dẫn tới thư mục gốc chứa dữ liệu ảnh.
- `--batch_size`: Số lượng samples trên mỗi batch (Mặc định: `8`).
- `--img_size`: Kích thước ảnh đầu vào để resize, giả định là hình vuông (Mặc định: `256`).
- `--num_workers`: Số luồng CPU dùng cho việc load data (Mặc định: `4`).
- `--n_epochs`: Tổng số vòng lặp huấn luyện (Mặc định: `50`).
- `--lr`: Tốc độ học (Learning Rate) của optimizer (Mặc định: `1e-4`).
- `--device`: Thiết bị chạy huấn luyện (`cpu` hoặc `gpu`. Mặc định: `cpu`).
- `--criterion`: Hàm mất mát sử dụng (`BCELoss` mặc định, có thể chọn `DiceLoss` hoặc `BCEDiceLoss`).
- `--model_save_path`: Nơi xuất file `.pth` (weight của model). Chú ý nếu không điền tham số này thì model sẽ không được lưu.
- `--Wavelet_Level`: Cấu hình cấp độ Wavelet áp dụng riêng cho model DTCWTSC-UNet (Mặc định: `1`). 

### 2. Cách Kiểm thử và Suy luận (Inference)

Sau khi lưu thành công trọng số (`.pth`), sử dụng file `inference.py` để chạy dự đoán trực quan, in ảnh cho các batch kết quả và tính các metrics đánh giá độ chính xác (Dice, IoU, Hausdorff, v.v.).

**Lệnh mẫu:**
```bash
cd Wavelet-based-Skip-Connections-U-Net

python -m inference --model_type DTCWTSC_UNet --checkpoint_path Models/my_model.pth --image_dir Datasets/Kvasir-SEG/images --mask_dir Datasets/Kvasir-SEG/masks --device gpu --img_size 256 --Wavelet_Level 1
```

**Giải thích chi tiết các tham số:**
- `--model_type` *(bắt buộc)*: Phải khớp với cấu trúc kiến trúc mô hình đã train (`Unet` hoặc `DTCWTSC_UNet`).
- `--checkpoint_path` *(bắt buộc)*: Đường dẫn đến file `.pth` chứa trọng số huấn luyện.
- `--image_dir` *(bắt buộc)*: Thư mục chứa các ảnh cần đưa vào suy luận để lấy ảnh dự đoán.
- `--mask_dir` *(bắt buộc)*: Thư mục chứa Ground Truth mask khớp với test set nhằm so sánh độ đo.
- `--device`: Thiết bị chạy tính toán (`cpu` hoặc `gpu`. Mặc định: `gpu`).
- `--img_size`: Kích thước ảnh resize phải tương đương lúc train (Mặc định: `256`).
- `--Wavelet_Level`: Cấp độ Wavelet sử dụng, cũng phải khớp với cấu hình lúc huấn luyện (Mặc định: `1`).
