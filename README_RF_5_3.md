# Chạy Random Forest cho tiểu mục 5.3

Nguyễn Ngọc Tuấn Anh — Nhóm 8 — IS403.R14

Gói này có đủ dữ liệu đã gửi, code dùng chung của TV2, chương trình RF, notebook, kết quả thực tế, hình minh chứng và báo cáo Word. Đọc kết quả chính trong `reports/Bao_cao_5_3_Random_Forest.docx`. Hai file `common/features.py` và `common/backtest.py` được sao chép nguyên bản từ file đã gửi, không dựng lại một pipeline đặc trưng riêng.

## 1 Chuẩn bị trên máy Windows

Clone repo và chuyển sang branch `feature-arima`, hoặc tải branch này vào thư mục riêng, ví dụ `D:\RF_5_3`. Bên trong thư mục có `run_rf.py`, `common`, `models`, `data`, `results`, `figures`, `reports` và notebook. Không copy đè toàn bộ vào repo nhóm trước khi kiểm tra.

Mở **Anaconda Prompt** và tạo môi trường riêng để giữ nguyên môi trường `dataanalysis` đang dùng cho các bài khác:

```bat
conda create -n rf53 python=3.12 -y
conda activate rf53
cd /d D:\RF_5_3
python -m pip install -r requirements_rf.txt
```

Thay `D:\RF_5_3` bằng đường dẫn thật tới thư mục chứa `run_rf.py`. Requirements ghi đúng các phiên bản đã chạy. XGBoost không bắt buộc vì chỉ chạy RF; `common/backtest.py` có import tùy chọn XGBoost và vẫn hoạt động khi không cài.

## 2 Cách chạy nhanh toàn bộ

```bat
python run_rf.py --stage all
```

Lệnh này kiểm tra dữ liệu, ghi thiết kế RF trước khi CV, chạy 24 lượt fit validation, khóa cấu hình, fit cuối và dự báo test, kiểm tra tích hợp, xuất CSV và hình. Kết quả tạo trong `results/` và `figures/`. Chạy lại sẽ thay các file kết quả và thời điểm khóa cấu hình; giữ bản ZIP gốc để đối chiếu. Không sửa grid hoặc đặc trưng sau khi xem test nhằm tìm điểm đẹp hơn.

## 3 Cách chạy từng bước để chụp minh chứng

**Bước A — Kiểm tra đầu vào**

```bat
python run_rf.py --stage prepare
```

Chụp dòng 83 tháng, train = 71, test = 12, supervised = 64 và bảng 40/46/52/64. Lệnh tạo `protocol_rf.json`, ghi thứ tự ưu tiên trước validation. Dòng hash của protocol gốc không khớp là phát hiện nguồn đầu vào, không phải lỗi chạy RF; xem mục 7.

**Bước B — Chọn cấu hình**

```bat
python run_rf.py --stage cv
```

Chụp bảng `VALIDATION SUMMARY` có đủ 8 cấu hình, RMSE cả 3 fold và trung bình; chụp dòng `LOCKED`. File `best_rf.json` được ghi ở đây, trước bước đánh giá test. Không dùng test chọn mô hình.

Quy tắc đã cố định: giữ các cấu hình trong 1% mức mean CV RMSE thấp nhất; ưu tiên `max_depth` nhỏ hơn, rồi `min_samples_leaf` lớn hơn; cùng hai tiêu chí thì lấy mean CV RMSE thấp hơn, cuối cùng mã cấu hình. Không mặc định max_features = 0.7 là đơn giản hơn 1.0.

**Bước C — Dự báo và kiểm tra**

```bat
python run_rf.py --stage forecast
```

Chụp bảng 12 dự báo, `RF TEST METRICS`, bảng đối chiếu ARIMA và `INTEGRATION CHECKS`. CSV lưu giá trị số chưa làm tròn; console chỉ định dạng để đọc. Không dùng ảnh làm nguồn nhập số cho TV6.

**Bước D — Xuất hình**

```bat
python run_rf.py --stage figures
```

Mở trực tiếp ảnh PNG trong `figures/` để chèn vào Word. Ảnh xuất trực tiếp nét hơn chụp toàn bộ IDE. Tối thiểu chèn `rf_forecast.png`; bảng validation có thể dùng bảng trong Word hoặc `rf_validation_table.png`.

Nếu nhận thông báo code/dữ liệu/protocol thay đổi, kiểm tra nguyên nhân trước. Chỉ chạy lại từ prepare khi có thay đổi kỹ thuật chính đáng và ghi rõ thay đổi; không bỏ kiểm tra hash để tiếp tục dùng cấu hình cũ.

## 4 Chạy notebook trong VS Code hoặc Jupyter

Notebook `notebooks/RF_5_3_TuanAnh.ipynb` chia rõ kiểm tra dữ liệu, validation, bảng chọn cấu hình, dự báo, chỉ số và hình. Mặc định `RUN_EXPERIMENT = False` để xem kết quả thực tế đã lưu; đổi thành `True` để tự huấn luyện lại. Có thể dùng notebook thay các lệnh theo stage; không cần chạy cả hai cách chỉ để lặp kết quả.

- VS Code: cài extension Python và Jupyter, mở thư mục gói, chọn kernel môi trường `rf53`.
- Nếu kernel chưa có: chạy `python -m pip install ipykernel`, sau đó chọn interpreter đúng môi trường.
- JupyterLab nếu muốn: `python -m pip install jupyterlab`, rồi `jupyter lab`.
- Chạy lần lượt từ trên xuống. Các ô tính toán quan trọng đã có kết quả thực thi trong bản notebook được giao; khi chạy lại, output được cập nhật từ máy của mình.

Kiểm tra kỹ thuật: chương trình CLI đã chạy thành công toàn bộ thực nghiệm. Các ô code notebook đã được thực thi bằng IPython trong cùng tiến trình ở chế độ đọc kết quả, và lưu output thực tế. Chưa xác minh được việc khởi động kernel Jupyter trong môi trường tạo gói do hạn chế socket; trên máy Windows, nếu kernel gặp vấn đề, có thể dùng các lệnh CLI ở mục 2–3. File Word là báo cáo của lần chạy được giao, không tự cập nhật khi thay dữ liệu hoặc chạy mô hình mới.

## 5 Hiểu mô hình để trình bày

Mình chỉ phụ trách **RandomForestRegressor**, không cần tự làm thêm SVR/XGBoost/LSTM. Tám cấu hình là 2 lựa chọn max_depth × 2 min_samples_leaf × 2 max_features, cố định 300 cây và seed 42.

RF học **tăng trưởng log đã chuẩn hóa z**, không học trực tiếp số xe. Với mỗi fold, nhãn r được chuẩn hóa bằng mean và std (ddof = 0) trên đúng các cặp học trong train. X gồm 5 đặc trưng quá khứ và không chuẩn hóa cho RF. Sau khi dự báo z, đổi về r rồi tích lũy về số xe:

`r_pred = mu_train + std_train * z_pred`

`y_pred[h] = y_origin * exp(sum(r_pred từ 1 đến h))`

Từ tháng dự báo thứ hai, lag/rolling dùng dự báo r trước đó, không dùng actual của test. RF học trực tiếp mức xe bằng trung bình ở lá khó vượt trần nhãn train; học r giúp quy mô tích lũy có thể vượt trần này, nhưng không bảo đảm RF thắng ARIMA.

**CV và test khác nhau:** CV dùng các giai đoạn nằm trong 71 tháng train để chọn cấu hình. Test là cửa sổ cố định 12/2022–11/2023 để đối chiếu cuối cùng. RMSE CV thấp hơn RMSE test không tự chứng minh có lỗi; đây là các giai đoạn và tốc độ tăng trưởng khác nhau.

**Bias = mean(actual − pred):** dương là dự báo thiếu; âm là dự báo dư. sMAPE không phải MAPE. Không lấy 100 − MAPE/sMAPE để gọi là accuracy. RF không cần AIC/BIC và không cần chép lại toàn bộ ADF/Ljung–Box của ARIMA.

## 6 File bàn giao

| File | Nội dung |
|---|---|
| `models/random_forest.py` | Grid RF, quy tắc chọn và wrapper dùng common API |
| `run_rf.py` | Chương trình chạy từng bước hoặc toàn bộ |
| `results/cv_rf.csv` | Đủ 24 dòng cấu hình × fold, có trạng thái và cảnh báo |
| `results/cv_rf_summary.csv` | Bảng 8 cấu hình và mean CV RMSE |
| `results/cv_forecasts_rf.csv` | 288 dự báo validation để đối chiếu |
| `results/best_rf.json` | Cấu hình khóa trước test, quy tắc chọn và phiên bản |
| `results/forecast_rf.csv` | Đúng schema nhóm, 12 dự báo chưa làm tròn |
| `results/trace_rf.csv` | Đặc trưng và dự báo z/r/y ở 12 bước |
| `results/metrics_rf.csv` | RMSE, MAE, sMAPE, Bias, số mẫu và thời gian fit cuối |
| `results/fit_metadata_rf.json` | Tham số estimator, scaler nhãn, nguồn dữ liệu và cảnh báo |
| `results/comparison_rf_arima.csv` | Điểm ARIMA tính lại từ CSV được gửi và đối chiếu RF |
| `results/integration_checks_rf.csv` | Các kiểm tra tích hợp RF đã chạy; không phải test_leakage.py gốc |
| `figures/rf_validation.png` | Biến động RMSE qua 3 fold |
| `figures/rf_validation_table.png` | Bảng validation dùng làm minh chứng |
| `figures/rf_forecast.png` | Hình thực tế và dự báo RF |
| `figures/rf_errors.png` | Sai số có dấu theo chân trời |
| `reports/Bao_cao_5_3_Random_Forest.docx` | Báo cáo đã điền kết quả thực nghiệm |

## 7 Nguồn và điểm cần thống nhất với nhóm

1. `protocol_original.json` được giữ nguyên. SHA256 của CSV được gửi không trùng hash trong protocol gốc. Các kiểm tra tháng, tổng thành phần và split đều cần vượt qua trước khi chạy; lần này sử dụng chính CSV được gửi, với hash thực lưu trong `protocol_rf.json`. Không khẳng định đã chứng minh trùng từng giá trị với phiên bản dữ liệu mang hash cũ. Nhóm cần xác nhận khi tích hợp chính thức; chưa cần thêm file để tái lập riêng RF trên CSV đã gửi.
2. Quy tắc đơn giản hơn trong 1% được cụ thể hóa cho RF trước khi chạy và ghi trong `protocol_rf.json`; protocol gốc chỉ nêu nguyên tắc chung.
3. `forecast_arima.csv` là nguồn đối chứng được gửi. Các chỉ số ARIMA trong báo cáo RF được tính lại từ chính 12 dự báo chưa làm tròn này, không ghép số từ các bảng khác trong PDF.
4. Chưa có forecast Ridge và SARIMA chính thức, nên không kết luận RF tốt/kém hơn hai mô hình đó. Gửi thêm các file này nếu muốn bổ sung so sánh; phần RF hiện đã có thể chạy đầy đủ.
5. Không nhận được test_leakage.py gốc. Gói có các kiểm tra tích hợp RF riêng, kiểm tra scaler nhãn, đặc trưng, trace đệ quy, đổi ngược và bất biến khi đổi giá trị tương lai. Kết quả này chỉ hỗ trợ các điều kiện được kiểm tra, không phải bảo đảm tuyệt đối không thể có lỗi trong mọi trường hợp.
6. Mục 5.1 trong PDF định nghĩa dấu Bias ngược với 5.6; các kết quả ở đây thống nhất theo actual − pred của common API được gửi.

Tài liệu bổ sung: scikit-learn RandomForestRegressor https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html và ví dụ lagged features https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html. Thiết kế chính theo phân công Chương 5 của Nhóm 8 và API từ các file đính kèm.
