# Quy trình tiền xử lý dữ liệu – Bộ dữ liệu [P] (Electric Vehicle Population Data, Bang Washington)

- **Mã nguồn:** [`src/preprocess_P.py`](../src/preprocess_P.py) · **Lệnh thực thi:** `python src/preprocess_P.py [--strict]`
- **Dữ liệu đầu vào:** `data/EV_Population_Washington_Clean__1_.csv` (134.726 dòng × 11 cột)
- **Dữ liệu đầu ra:** `data/processed/EV_Population_WA_Processed.csv` (134.726 dòng × 15 cột) và nhật ký xử lý `results/P_preprocessing_audit.csv`

**Nguyên tắc thiết kế thừa hưởng từ Báo cáo Nhóm 8 (Mục 3.2):**
- [P] là **dữ liệu cắt ngang (cross-sectional)** phản ánh ảnh chụp đội xe (không có mốc ngày tháng biến động).
- Giá trị thiếu của `Electric Range` / `Base MSRP` **tuyệt đối không thay bằng 0** và không điền khuyết (imputation) tùy tiện; giữ nguyên giá trị thiếu (NA).
- **Không xóa bỏ dòng trùng**: do file làm sạch không còn cột `DOL Vehicle ID`, không thể xác định đó là cùng một xe hay các xe cùng loại trong cùng khu vực; chỉ đếm và ghi nhận.
- Giữ nguyên các tên cột gốc để đảm bảo tính tương thích với các script kiểm định giả thuyết (`src/hypothesis_tests_P.py`); chỉ **bổ sung thêm** các cột đặc trưng phái sinh.

```mermaid
flowchart LR
  A[1 Đọc dữ liệu] --> B[2 Kiểm tra Schema] --> C[3 Lọc phạm vi WA] --> D[4 Chuẩn hóa văn bản] --> E[5 Ép kiểu dữ liệu]
  E --> F[6 Xử lý giá trị bất hợp lệ] --> G[7 Tạo biến phái sinh] --> H[8 Đánh giá giá trị thiếu]
  H --> I[9 Đánh giá bản ghi trùng] --> J[10 Đối chiếu báo cáo] --> K[11 Lưu trữ kết quả]
```

| STT | Bước xử lý | What (Làm gì?) | How (Làm như thế nào?) | Why (Tại sao phải làm?) | Kết quả thực tế trên dữ liệu |
|:---:|------------|----------------|------------------------|-------------------------|------------------------------|
| **1** | **Đọc dữ liệu (Load)** | Đọc file CSV vào DataFrame | Sử dụng `pd.read_csv`, chỉ định `Postal Code` dạng chuỗi (`string`) | File CSV lưu mã ZIP dưới dạng số thực `99344.0`; ép kiểu text ngay từ đầu giúp bảo toàn mã và tránh tự suy diễn sai kiểu dữ liệu | 134.726 dòng, 11 cột |
| **2** | **Kiểm tra Schema** | Xác minh cấu trúc 11 cột chuẩn theo báo cáo | So sánh danh sách cột với `EXPECTED_COLUMNS`, báo lỗi nếu thiếu cột | Phát hiện lỗi sớm nếu file dữ liệu đầu vào bị sai; đảm bảo các bước tính toán phía sau luôn tìm thấy đúng cột | Đủ 11 cột chuẩn; không có cột dư thừa |
| **3** | **Lọc phạm vi (Scope)** | Chỉ giữ các xe đăng ký tại bang Washington (`State == WA`) | Lọc theo mặt nạ boolean sau khi đã cắt khoảng trắng và viết hoa | Phạm vi nghiên cứu chỉ tập trung vào Bang Washington (Mục 1.4 & 3.2). Đồng thời là chốt an toàn nếu áp dụng cho file thô 17 cột | Giữ nguyên 134.726 dòng (0 dòng bị loại) |
| **4** | **Chuẩn hóa văn bản (Text)** | Loại bỏ khoảng trắng thừa, đồng nhất chữ hoa/thường, chuẩn hóa tên Thành phố | Regex chuẩn hóa khoảng trắng; viết hoa toàn bộ `Make` & `Model`; hợp nhất biến thể chính tả `City` (`McCleary`) | Tránh việc cùng một danh mục bị tách thành nhiều nhóm (ảnh hưởng đến xếp hạng Top-N và kiểm định $\chi^2$ H3, H4a) | Chuẩn hóa 21 dòng của `City`; số lượng City duy nhất: 454 → 453 |
| **5** | **Ép kiểu dữ liệu (Types)** | Chuẩn hóa `Postal Code` về chuỗi 5 ký tự; chuyển `Model Year`, `Electric Range`, `Base MSRP` sang số nguyên nullable (`Int64`) | Sử dụng `to_numeric` $\to$ `.astype("Int64")`; dùng `.str.zfill(5)` cho mã ZIP | Định dạng CSV lưu số nguyên thành dạng float (`2020.0`); dùng kiểu `Int64` để vừa hiển thị đúng số nguyên vừa bảo toàn các ô thiếu (NA) | Hoàn thành; các cột số được chuyển thành `Int64` |
| **6** | **Xử lý giá trị bất hợp lệ** | Chuyển các giá trị phi lý / placeholder sang NA: `Range` $\le 0$ hoặc $> 500$; `MSRP` $\le 0$; `Model Year` ngoài khoảng 1980..nay | Áp dụng mặt nạ boolean và `.mask()`; ghi nhận vào nhật ký kiểm toán | DOL dùng `0` cho các xe chưa nghiên cứu tầm pin. Nếu giữ `0` sẽ kéo tụt trung bình và làm sai lệch kiểm định phi tham số H1, H2 | 0 dòng vi phạm; 37 xe đời trước 2010 được giữ nguyên vì là xe hợp lệ |
| **7** | **Tạo biến phái sinh (Features)** | Bổ sung 4 cột mới: `Type` (BEV/PHEV), `CAFV Status`, `Range Missing` (boolean), `County City` | Regex trích xuất, ánh xạ từ điển, `.isna()`, ghép chuỗi | `Type` là biến phân nhóm chính cho H1-H4; `County City` cần thiết vì có 38 tên thành phố xuất hiện ở nhiều hơn 1 hạt (493 cặp) | Thêm 4 cột: `Type`, `CAFV Status`, `Range Missing`, `County City` |
| **8** | **Đánh giá giá trị thiếu** | Thống kê tỷ lệ thiếu theo từng cột và theo loại xe; **không điền khuyết (imputation)** | Sử dụng `.isna().sum()`, `groupby("Type")`, lập bảng chéo với CAFV | Bản chất thiếu là do cơ chế (BEV đời $\ge 2021$ chưa được nghiên cứu tầm pin $\leftrightarrow$ CAFV Unknown). Điền giá trị sẽ gây méo mó dữ liệu | `Electric Range` thiếu 56.869 (42,21%); `Base MSRP` thiếu 131.307 (97,46%); `Model` thiếu 249 (0,18%) |
| **9** | **Đánh giá bản ghi trùng** | Thống kê số lượng dòng giống nhau trên 11 cột; **không thực hiện xóa dòng** | Hàm `.duplicated(subset=...)` và đếm quy mô nhóm trùng lớn nhất | File không chứa mã định danh xe (`DOL Vehicle ID`); các xe cùng dòng/đời/khu vực hoàn toàn có thể trùng thuộc tính; xóa dòng sẽ làm mất 70% dữ liệu | Phát hiện 93.707 dòng (69,6%) lặp thuộc tính; giữ nguyên toàn bộ 134.726 bản ghi |
| **10** | **Đối chiếu báo cáo** | So khớp tự động 11 chỉ số tổng hợp với số liệu đã công bố trong báo cáo | Lập từ điển đối chiếu, so sánh từng cặp giá trị; cờ `--strict` sẽ ngắt nếu sai lệch | Đảm bảo tính nhất quán tuyệt đối giữa mã tiền xử lý và nội dung trình bày trong báo cáo nghiên cứu | Khớp chính xác 11/11 chỉ số (100%) |
| **11** | **Lưu trữ dữ liệu** | Ghi file dữ liệu đã xử lý và xuất file nhật ký kiểm toán (audit log) ra đĩa | Sử dụng `.to_csv(..., encoding="utf-8-sig")` | Cung cấp đầu vào chuẩn hóa, nhất quán cho EDA và kiểm định giả thuyết; nhật ký kiểm toán là minh chứng cho Chương 3 | Đã lưu: `EV_Population_WA_Processed.csv` và `P_preprocessing_audit.csv` |

---

## Các phát hiện quan trọng phục vụ báo cáo

1. **Số lượng Thành phố (City):**
   - Trong báo cáo ghi nhận 454 City theo tên chuỗi thô ban đầu.
   - Sau khi chuẩn hóa các biến thể viết hoa/thường (`Mccleary` và `McCleary`), số City thực tế là **453**. Tổng số cặp định danh `(County, City)` vẫn được bảo toàn nguyên vẹn là **493**.
   - *Đề xuất cho báo cáo:* Bổ sung chú thích *"454 tên chuỗi thô, tương ứng 453 thành phố chuẩn hóa sau khi đồng nhất chữ hoa/thường"*.

2. **Trường hợp Thành phố McCleary:**
   - Xuất hiện ở 2 County: Mason (1 xe) và Grays Harbor (21 xe). Đây là hiện tượng lỗi nhập liệu thực tế từ dữ liệu gốc, quy trình giữ nguyên tính khách quan của dữ liệu và không tự ý sửa đổi vị trí hành chính.

3. **Giải đáp cho các ghi chú phản biện (Comment C1, C2, C3 trong bản thảo):**
   - **C1 (Lọc State khác WA):** Trên file làm sạch hiện tại, 100% dòng đều có `State == WA` (0 dòng bị loại). Số dòng bị loại ở bước thô ban đầu chỉ có thể truy xuất từ file gốc 17 cột.
   - **C2 (Giá trị 0 của Range & MSRP):** Trên file sạch không còn giá trị `0` nào (đã được chuyển thành NA sẵn từ trước).
   - **C3 (Bản ghi trùng):** Không có cột `DOL Vehicle ID` trên file 11 cột để kiểm tra trùng xe thực tế. Toàn bộ 134.726 dòng được bảo toàn nguyên vẹn vì mỗi dòng đại diện cho một phương tiện đăng ký hợp lệ.
