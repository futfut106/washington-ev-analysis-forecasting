# Washington EV Analysis & Forecasting

**Phân tích đặc điểm, xu hướng phân bố và dự báo số lượng ô tô điện (BEV và PHEV) tại Washington, Hoa Kỳ**

## 1. Giới thiệu dự án

Dự án tập trung phân tích dữ liệu ô tô điện tại bang Washington, Hoa Kỳ, nhằm tìm hiểu đặc điểm phương tiện, sự phân bố theo khu vực địa lý và xu hướng thay đổi số lượng xe điện theo thời gian.

Đối tượng nghiên cứu bao gồm:

- **BEV (Battery Electric Vehicle):** Ô tô điện chạy hoàn toàn bằng pin.
- **PHEV (Plug-in Hybrid Electric Vehicle):** Ô tô hybrid có thể sạc pin từ nguồn điện bên ngoài.

Thông qua Python và các phương pháp phân tích dữ liệu, dự án hướng đến việc khám phá dữ liệu, trực quan hóa các đặc điểm nổi bật, xây dựng mô hình dự báo chuỗi thời gian và đánh giá hiệu quả của các phương pháp dự báo.

## 2. Mục tiêu nghiên cứu

- Phân tích đặc điểm xe điện theo loại phương tiện, hãng sản xuất, mẫu xe và năm sản xuất.
- Khảo sát sự phân bố xe điện theo quận (county) và thành phố (city) tại Washington.
- Phân tích xu hướng phát triển số lượng xe điện qua các tháng và năm.
- Xây dựng và so sánh các mô hình dự báo số lượng xe điện dựa trên dữ liệu lịch sử.
- Đánh giá hiệu quả mô hình bằng các chỉ số phù hợp và trực quan hóa kết quả.
- Tổng hợp kết quả thành các bảng, biểu đồ và báo cáo phục vụ nghiên cứu.

## 3. Cấu trúc repository

```text
washington-ev-analysis-forecasting/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
├── notebooks/
├── src/
├── results/
├── figures/
└── reports/
```

### 3.1. `data/` – Dữ liệu nghiên cứu

Lưu trữ các bộ dữ liệu đầu vào và dữ liệu đã qua xử lý được sử dụng trong dự án.

Nội dung dự kiến:
- Dữ liệu về đặc điểm và số lượng xe điện tại Washington.
- Dữ liệu lịch sử số lượng xe điện theo thời gian.
- Dữ liệu đã được làm sạch, chuẩn hóa và tổng hợp phục vụ phân tích.

Các tệp dữ liệu cần được đặt tên rõ ràng và có thông tin về nguồn dữ liệu, phạm vi thời gian và ý nghĩa các thuộc tính. Dữ liệu đầu vào cần được giữ nguyên khi cần đối chiếu với dữ liệu sau xử lý.

### 3.2. `notebooks/` – Notebook phân tích

Lưu trữ các Jupyter Notebook được sử dụng để thực hiện và ghi lại quy trình phân tích dữ liệu.

Nội dung dự kiến:
- Khám phá dữ liệu và thống kê mô tả (EDA).
- Phân tích đặc điểm và phân bố xe điện.
- Trực quan hóa xu hướng số lượng xe điện theo thời gian.
- Thử nghiệm các phương pháp dự báo.
- Đánh giá, so sánh mô hình và trình bày kết quả.

Notebook có thể được thực thi trên Google Colab hoặc môi trường Jupyter Notebook tương thích.

### 3.3. `src/` – Mã nguồn Python

Chứa các chương trình và hàm Python được xây dựng để xử lý dữ liệu, phân tích và dự báo.

Nội dung dự kiến:
- Đọc và tiền xử lý dữ liệu.
- Tổng hợp dữ liệu theo các tiêu chí nghiên cứu.
- Xây dựng đặc trưng đầu vào cho mô hình.
- Huấn luyện và đánh giá các mô hình dự báo.
- Xuất bảng kết quả và biểu đồ.

Việc tổ chức mã nguồn thành các mô-đun riêng giúp hạn chế lặp mã, thuận tiện cho việc bảo trì và tái sử dụng.

### 3.4. `results/` – Kết quả phân tích và dự báo

Lưu trữ các bảng kết quả được tạo ra trong quá trình thực hiện dự án.

Nội dung dự kiến:
- Bảng thống kê mô tả và dữ liệu tổng hợp.
- Kết quả dự báo số lượng xe điện.
- Bảng so sánh giá trị thực tế và giá trị dự báo.
- Các chỉ số đánh giá mô hình như MAE, RMSE và MAPE.
- Các tệp CSV hoặc Excel phục vụ tổng hợp kết quả.

Mỗi kết quả cần được ghi nhận rõ phương pháp thực hiện và giai đoạn dữ liệu sử dụng để thuận tiện cho việc kiểm tra và đối chiếu.

### 3.5. `figures/` – Biểu đồ trực quan hóa

Lưu trữ các biểu đồ được tạo ra từ quá trình phân tích và đánh giá mô hình.

Nội dung dự kiến:
- Biểu đồ phân bố xe điện theo quận và thành phố.
- Biểu đồ so sánh số lượng BEV và PHEV.
- Biểu đồ theo hãng sản xuất, mẫu xe và năm sản xuất.
- Biểu đồ xu hướng số lượng xe điện theo thời gian.
- Biểu đồ so sánh giá trị thực tế với giá trị dự báo.

Các biểu đồ sử dụng trong báo cáo nên có tiêu đề, nhãn trục, đơn vị đo và chú thích phù hợp.

### 3.6. `reports/` – Báo cáo nghiên cứu

Lưu trữ các tài liệu trình bày quá trình nghiên cứu, phương pháp thực hiện và kết quả dự án.

Nội dung dự kiến:
- Báo cáo nghiên cứu chính thức.
- Tài liệu mô tả dữ liệu và phương pháp phân tích.
- Slide thuyết trình.
- Phụ lục, tài liệu tham khảo và các tài liệu liên quan.

Các nhận xét và kết luận trong báo cáo cần dựa trên kết quả thực nghiệm đã được kiểm tra.

### 3.7. `requirements.txt` – Thư viện cần thiết

Liệt kê các thư viện Python cần thiết để chạy mã nguồn trong dự án.

Các thư viện chính dự kiến bao gồm:

| Thư viện | Mục đích |
|---|---|
| `pandas` | Đọc và xử lý dữ liệu dạng bảng |
| `numpy` | Tính toán số học |
| `matplotlib` | Trực quan hóa dữ liệu |
| `scikit-learn` | Xây dựng và đánh giá mô hình học máy |
| `statsmodels` | Phân tích thống kê và chuỗi thời gian |
| `openpyxl` | Đọc và ghi tệp Excel |

Cài đặt các thư viện bằng lệnh:

```bash
pip install -r requirements.txt
```

### 3.8. `.gitignore` – Các tệp không đưa lên GitHub

Xác định các tệp và thư mục không cần được theo dõi bởi Git, chẳng hạn như bộ nhớ đệm Python, môi trường ảo, tệp tạm và thông tin cấu hình riêng tư.

Điều này giúp repository gọn gàng, tránh đưa các tệp phát sinh không cần thiết vào lịch sử quản lý mã nguồn.

## 4. Công nghệ sử dụng

- **Python:** Ngôn ngữ lập trình chính.
- **Pandas và NumPy:** Tiền xử lý và phân tích dữ liệu.
- **Matplotlib:** Xây dựng biểu đồ trực quan hóa.
- **Scikit-learn:** Xây dựng và đánh giá mô hình học máy.
- **Statsmodels:** Phân tích thống kê và mô hình chuỗi thời gian.
- **Jupyter Notebook / Google Colab:** Thực hiện và ghi lại quá trình phân tích.
- **Git và GitHub:** Quản lý mã nguồn, lưu trữ tài liệu và theo dõi tiến độ dự án.

## 5. Quy trình thực hiện

1. Thu thập và tìm hiểu nguồn dữ liệu.
2. Kiểm tra chất lượng, làm sạch và chuẩn hóa dữ liệu.
3. Thực hiện phân tích khám phá dữ liệu và thống kê mô tả.
4. Phân tích đặc điểm và xu hướng phân bố xe điện.
5. Xây dựng các mô hình dự báo dựa trên dữ liệu lịch sử.
6. Đánh giá và so sánh mô hình bằng các chỉ số phù hợp.
7. Trực quan hóa kết quả và tổng hợp báo cáo nghiên cứu.

## 6. Cách bắt đầu

### Bước 1: Tải repository

```bash
git clone https://github.com/futfut106/washington-ev-analysis-forecasting.git
cd washington-ev-analysis-forecasting
```

### Bước 2: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Bước 3: Chuẩn bị dữ liệu

Đặt các bộ dữ liệu cần sử dụng vào thư mục `data/`. Kiểm tra tên tệp, đường dẫn và cấu trúc cột trước khi chạy notebook hoặc mã nguồn.

### Bước 4: Thực hiện phân tích

Mở các notebook trong thư mục `notebooks/` bằng Jupyter Notebook hoặc tải lên Google Colab. Thực hiện các bước theo thứ tự và kiểm tra kết quả sau mỗi giai đoạn.

### Bước 5: Lưu kết quả

Lưu các bảng kết quả vào `results/`, biểu đồ vào `figures/` và báo cáo vào `reports/`.

## 7. Lưu ý về dữ liệu và kết quả

Kết quả phân tích và dự báo phụ thuộc vào chất lượng dữ liệu, phạm vi thời gian và phương pháp được lựa chọn. Cần phân biệt rõ dữ liệu đầu vào, dữ liệu sau xử lý, kết quả thực nghiệm và giá trị dự báo.

Các mô hình cần được đánh giá trên tập dữ liệu kiểm tra phù hợp với thứ tự thời gian. Những kết luận về xu hướng phát triển hoặc khả năng dự báo cần được hỗ trợ bởi kết quả phân tích thực tế.

## 8. Trạng thái dự án

Repository được sử dụng để tổ chức dữ liệu, mã nguồn, kết quả thực nghiệm và tài liệu cho đề tài nghiên cứu. Nội dung sẽ được cập nhật trong quá trình triển khai.

**Nguồn dữ liệu và tài liệu tham khảo:** Sẽ được bổ sung cùng đường dẫn nguồn tương ứng sau khi được xác minh và sử dụng trong nghiên cứu.
# washington-ev-analysis-forecasting