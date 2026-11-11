# Chương 5 · TV1 Trần Thế Anh

| Tệp | Nội dung |
| --- | --- |
| ch4_series.csv | Chuỗi toàn bang 83 tháng, 71 train và 12 test. |
| ridge.py / Ridge_TV1.ipynb | Mã và notebook chạy Ridge theo ba fold. |
| protocol.json | Mốc chia, năm đặc trưng và SHA-256 dữ liệu. |
| cv_ridge.csv / best_ridge.json | Điểm của ba alpha và cấu hình được chọn. |
| forecast_ridge.csv / trace_ridge.csv | 12 dự báo chưa làm tròn và đặc trưng theo bước. |
| ridge_test_metrics.json | Điểm kiểm tra của Ridge. |
| forecast_ch4_baselines.csv | Dự báo Naive và Drift trên cùng kỳ kiểm tra. |
| ch4_results_manifest.csv | Nguồn và trạng thái đối chiếu các mốc Chương 4. |

Chạy từ thư mục này:

    python -m pip install numpy pandas
    python ridge.py --input ch4_series.csv --output results

Ridge chọn alpha = 10. RMSE test 5.325,82 xe, MAE 4.123,90 xe, sMAPE 2,85%. ARIMA/SARIMA là mốc tham chiếu trong khi chờ CSV dự báo chưa làm tròn. Tiểu kết toàn chương cần kết quả tổng hợp của TV6.
