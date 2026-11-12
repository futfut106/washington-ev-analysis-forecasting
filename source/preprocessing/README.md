# Tái dựng tiền xử lý Chương 3

Các script tái dựng một quy trình từ hai CSV raw đến hai CSV clean đã được cung cấp. Đây là mã mới cho quy trình tương đương được kiểm chứng, không phải mã gốc đã mất. Không suy luận rằng thứ tự câu lệnh lịch sử chắc chắn giống mã tái dựng.

## Cách chạy

Từ thư mục gốc repository, dùng Python có pandas:

```powershell
python source/preprocessing/run_chapter3.py
```

Đường dẫn đầu vào mặc định trỏ đến bốn file của người dùng trong `D:/Phân tích kinh doanh_2627`. Có thể thay đổi bằng `--population-raw`, `--history-raw`, `--population-reference`, `--history-reference`. Các file reference chỉ được sử dụng ở bước kiểm chứng, không tham gia tạo dữ liệu clean.

Kết quả được lưu tại `data/reconstructed_chapter3/`:

- `EV_Population_Washington_Clean.csv`: tái tạo Population.
- `EV_Population_History_Washington_Clean.csv`: tái tạo History ở cấp tháng–quận–nhóm sử dụng, chưa tổng hợp toàn bang.
- `History_ToanBang_TheoThang.csv`: dữ liệu phân tích dẫn xuất, 83 tháng, 8 cột, có nhãn train/test.
- `verification.json`: số liệu trước/sau, mã SHA-256, kết quả đối chiếu và các kiểm tra chất lượng.

Các file kết quả đã được tạo trong lần kiểm chứng. Để chạy lại, chọn một thư mục kết quả mới. Script từ chối ghi đè các file đã có:

```powershell
python source/preprocessing/run_chapter3.py --output-dir data/reconstructed_chapter3_run2
```

## Chạy riêng từng bước

Hai lệnh clean chỉ cần raw, không cần reference. Các đường dẫn output dưới đây phải chưa tồn tại:

```powershell
python source/preprocessing/clean_population.py --raw "D:/Phân tích kinh doanh_2627/archive/Electric_Vehicle_Population_Data.csv" --output "data/population_rerun.csv"
python source/preprocessing/clean_history.py --raw "D:/Phân tích kinh doanh_2627/Electric_Vehicle_Population_Size_History_By_County.csv/Electric_Vehicle_Population_Size_History_By_County.csv" --output "data/history_rerun.csv"
```

Kiểm chứng riêng một đầu ra:

```powershell
python source/preprocessing/verify_preprocessing.py --generated "data/population_rerun.csv" --reference "D:/Phân tích kinh doanh_2627/archive/EV_Population_Washington_Clean (1).csv"
```

## Điều kiện nghiệm thu

1. Đọc raw và áp dụng các biến đổi xác định trong từng script.
2. So sánh bảng được đọc lại từ đầu ra với reference, chính xác từng ô, vị trí thiếu, cột và thứ tự dòng; không dùng dung sai số thực.
3. So sánh SHA-256 toàn file; nếu cách xuất CSV khác reference, script cũng báo lỗi.
4. Kiểm tra lại mã băm của cả bốn file đầu vào để xác nhận chúng không bị thay đổi.
5. History phải đủ tổ hợp tháng × quận × mục đích sử dụng, không thiếu, không trùng khóa, ngày cuối tháng liên tục, thỏa các quan hệ số lượng và tỷ lệ.

Môi trường đã kiểm chứng thành công: Python 3.12.10, pandas 3.0.5. Định dạng xuất: UTF-8 có BOM, xuống dòng LF, không ghi index, giá trị thiếu là ô trống. Các cột số ở dạng float được giữ để tái tạo đúng reference; không ép năm và các cột đếm về int trong hai file clean.

## Phân biệt hai giai đoạn

Clean Population: lọc WA → giữ 11 cột → đổi 0 của Range/MSRP thành thiếu → biểu diễn số dạng float → sắp xếp County, City, Model Year, Make, giữ thứ tự nguồn khi trùng khóa → xuất CSV.

Clean History: lọc WA → chuẩn hóa ngày → biểu diễn số dạng float → sắp xếp Date, County, giữ thứ tự nguồn khi trùng khóa → xuất CSV.

Chuẩn bị phân tích: kiểm tra chất lượng History → cộng năm cột số lượng theo Date → tính lại tỷ lệ → chia 71 tháng train và 12 tháng test. Giai đoạn này không làm thay đổi hai file clean đã tái tạo.

Không điền khuyết, không thay Model trống bằng nhãn, không bỏ bản ghi có cùng thuộc tính, không chuẩn hóa nhãn có sẵn, không xóa ngoại lệ. Những thao tác đó không nằm trong quy trình đã được kiểm chứng.
