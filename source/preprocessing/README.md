# Tiền xử lý dữ liệu Chương 3

Thư mục này chứa các script Python xử lý hai bộ dữ liệu xe điện Washington từ file raw thành file clean và chuẩn bị dữ liệu cho phân tích, dự báo. Quy trình gồm lọc phạm vi nghiên cứu, lựa chọn thuộc tính, xử lý giá trị 0 và giá trị thiếu, chuẩn hóa định dạng, sắp xếp dữ liệu và kiểm tra chất lượng đầu ra.

## Dữ liệu đầu vào và đầu ra

| Bộ dữ liệu | File raw | File clean | Kích thước trước và sau xử lý |
|---|---|---|---|
| Population | `Electric_Vehicle_Population_Data.csv` | `EV_Population_Washington_Clean.csv` | 135.038 dòng × 17 cột → 134.726 dòng × 11 cột |
| History | `Electric_Vehicle_Population_Size_History_By_County.csv` | `EV_Population_History_Washington_Clean.csv` | 19.700 dòng × 10 cột → 6.474 dòng × 10 cột |

Population mô tả đặc điểm từng bản ghi phương tiện. History ghi nhận số lượng phương tiện theo tháng, quận và nhóm sử dụng. Phạm vi phân tích của cả hai bộ dữ liệu là bang Washington (`State = WA`).

## 1. Xử lý Population

Script: `clean_population.py`.

1. Đọc CSV raw và kiểm tra các cột cần thiết.
2. Giữ các dòng có `State = WA`, loại 312 dòng ngoài phạm vi nghiên cứu.
3. Giữ 11 cột phục vụ phân tích. Bỏ `VIN (1-10)`, `Legislative District`, `DOL Vehicle ID`, `Vehicle Location`, `Electric Utility` và `2020 Census Tract`.
4. Chuyển 56.868 giá trị 0 trong `Electric Range` và 131.306 giá trị 0 trong `Base MSRP` thành giá trị thiếu. Tính cả ô thiếu có sẵn, hai cột lần lượt có 56.869 và 131.307 ô thiếu.
5. Giữ nguyên 249 ô thiếu trong `Model`; giữ các dòng này cho những phân tích không yêu cầu thông tin mẫu xe.
6. Biểu diễn `Postal Code`, `Model Year`, `Electric Range` và `Base MSRP` dưới dạng số thực khi xuất CSV. `Postal Code` vẫn được hiểu là biến định danh; `Model Year` là năm mẫu xe.
7. Sắp xếp tăng dần theo `County`, `City`, `Model Year`, `Make`. Khi các khóa bằng nhau, giữ thứ tự xuất hiện trong dữ liệu nguồn.
8. Xuất CSV với mã hóa UTF-8 có BOM, xuống dòng LF, không ghi index; giá trị thiếu được ghi bằng ô trống.

Các bản ghi có cùng 11 thuộc tính được giữ lại vì có thể thuộc các phương tiện khác nhau. Quy trình giữ nguyên nhãn phân loại và các giá trị quan sát khác; không điền khuyết hoặc xóa ngoại lệ.

## 2. Xử lý History

Script: `clean_history.py`.

1. Đọc CSV raw và kiểm tra cấu trúc 10 cột.
2. Giữ các dòng có `State = WA`. Có 13.226 dòng không thỏa điều kiện, bao gồm 83 dòng thiếu mã bang.
3. Chuẩn hóa `Date` từ ngày dạng chữ, ví dụ `October 31 2019`, sang `2019-10-31`.
4. Biểu diễn sáu cột số liệu dưới dạng số thực khi xuất CSV, giữ nguyên giá trị số lượng và tỷ lệ. Các giá trị 0 trong cột đếm được giữ lại.
5. Sắp xếp tăng dần theo `Date`, `County`, giữ thứ tự nguồn khi hai khóa bằng nhau.
6. Xuất CSV cùng định dạng với Population.

History clean có đủ 39 quận, 83 tháng từ 01/2017 đến 11/2023 và hai nhóm `Passenger`, `Truck`, tương ứng 6.474 dòng. File này giữ cấp độ tháng–quận–nhóm sử dụng, trước bước tổng hợp toàn bang.

## 3. Kiểm tra chất lượng và chuẩn bị chuỗi dự báo

Script điều phối: `run_chapter3.py`.

History được kiểm tra giá trị thiếu, trùng khóa `Date`–`County`–`Vehicle Primary Use`, tính đầy đủ của các tổ hợp quan sát, tính liên tục của tháng và ngày cuối tháng. Các cột đếm phải không âm và có giá trị nguyên về mặt số học. Các quan hệ sau phải thỏa mãn:

- `BEV + PHEV = EV Total`.
- `EV Total + Non-Electric Vehicle Total = Total Vehicles`.
- `Percent Electric Vehicles = EV Total / Total Vehicles × 100`, đối chiếu sau khi làm tròn hai chữ số thập phân.

Sau kiểm tra, cộng năm cột số lượng theo `Date` để tạo chuỗi toàn bang. Mỗi tháng, 78 dòng chi tiết của 39 quận và hai nhóm sử dụng được tổng hợp thành một dòng. Tỷ lệ EV toàn bang được tính lại từ tổng số lượng, thay vì lấy trung bình đơn giản tỷ lệ của từng dòng.

Chuỗi có 83 dòng và tám cột: `date`, `bev`, `phev`, `ev_total`, `non_ev`, `total_vehicles`, `ev_pct`, `split`. Tập train gồm 71 tháng từ 01/2017 đến 11/2022; tập test gồm 12 tháng từ 12/2022 đến 11/2023. Dữ liệu được chia theo thời gian và không xáo trộn.

## 4. Cách chạy

Yêu cầu Python và pandas. Môi trường đã kiểm tra: Python 3.12.10, pandas 3.0.5.

Chạy từ thư mục gốc repository:

```powershell
python source/preprocessing/run_chapter3.py --output-dir data/processed_chapter3
```

Lệnh sử dụng các đường dẫn dữ liệu mặc định trong `D:/Phân tích kinh doanh_2627`. Có thể cấu hình đường dẫn bằng `--population-raw`, `--history-raw`, `--population-reference` và `--history-reference`. Hai tham số `reference` chỉ phục vụ đối chiếu đầu ra; các hàm làm sạch chỉ sử dụng dữ liệu raw.

Thư mục kết quả chứa:

- `EV_Population_Washington_Clean.csv`: Population sau xử lý.
- `EV_Population_History_Washington_Clean.csv`: History sau xử lý.
- `History_ToanBang_TheoThang.csv`: chuỗi toàn bang 83 tháng, có nhãn train/test.
- `verification.json`: thống kê xử lý, kết quả kiểm tra và mã SHA-256.

Script từ chối ghi đè file đã tồn tại. Khi chạy lại, chọn thư mục đầu ra mới bằng `--output-dir`.

### Chạy riêng từng bộ dữ liệu

Hai script dưới đây chỉ cần file raw và đường dẫn đầu ra chưa tồn tại:

```powershell
python source/preprocessing/clean_population.py --raw "D:/Phân tích kinh doanh_2627/archive/Electric_Vehicle_Population_Data.csv" --output "data/processed_chapter3/population.csv"
python source/preprocessing/clean_history.py --raw "D:/Phân tích kinh doanh_2627/Electric_Vehicle_Population_Size_History_By_County.csv/Electric_Vehicle_Population_Size_History_By_County.csv" --output "data/processed_chapter3/history.csv"
```

### Đối chiếu kết quả

Script `verify_preprocessing.py` so sánh từng giá trị, vị trí thiếu, thứ tự cột, thứ tự dòng và mã SHA-256 với file clean tham chiếu:

```powershell
python source/preprocessing/verify_preprocessing.py --generated "data/processed_chapter3/population.csv" --reference "D:/Phân tích kinh doanh_2627/archive/EV_Population_Washington_Clean (1).csv"
```

Kết quả kiểm tra trên hai bộ dữ liệu cho thấy các đầu ra khớp toàn bộ byte với file clean tham chiếu tương ứng. Quy trình điều phối cũng kiểm tra mã băm của bốn file đầu vào trước và sau khi chạy để xác nhận chúng được giữ nguyên.
