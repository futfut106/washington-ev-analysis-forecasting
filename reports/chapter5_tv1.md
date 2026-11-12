# Chương 5 · TV1 Trần Thế Anh

| Tệp | Nội dung |
| --- | --- |
| `data/ch4_series.csv` | Chuỗi toàn bang 83 tháng, 71 train và 12 test. |
| `src/ridge.py` / `notebooks/Ridge_TV1.ipynb` | Mã và notebook chạy Ridge theo ba fold. |
| `data/protocol.json` | Mốc chia, năm đặc trưng và SHA-256 dữ liệu. |
| `results/cv_ridge.csv` / `results/best_ridge.json` | Điểm của ba alpha và cấu hình được chọn. |
| `results/forecast_ridge.csv` / `results/trace_ridge.csv` | 12 dự báo chưa làm tròn và đặc trưng theo bước. |
| `results/ridge_test_metrics.json` | Điểm kiểm tra của Ridge. |
| `results/forecast_ch4_baselines.csv` | Dự báo Naive và Drift trên cùng kỳ kiểm tra. |
| `results/ch4_results_manifest.csv` | Nguồn và trạng thái đối chiếu các mốc Chương 4. |

Các đường dẫn trên tính từ thư mục gốc repository và tuân theo cấu trúc trong README chính.

Chạy từ thư mục gốc repository:

```bash
python -m pip install numpy pandas
python src/ridge.py --input data/ch4_series.csv --output results
```

Hoặc mở `notebooks/Ridge_TV1.ipynb` với thư mục làm việc là thư mục gốc repository hoặc `notebooks/`. Notebook đọc dữ liệu từ `data/` và ghi kết quả vào `results/` ở gốc repository.

Ridge chọn alpha = 10. RMSE test 5.325,82 xe, MAE 4.123,90 xe, sMAPE 2,85%. ARIMA/SARIMA là mốc tham chiếu trong khi chờ CSV dự báo chưa làm tròn. Tiểu kết toàn chương cần kết quả tổng hợp của TV6.
