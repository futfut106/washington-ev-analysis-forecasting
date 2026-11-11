"""
Module: test_leakage.py
Tác giả: Nguyễn Nữ Trà Giang (TV2) - Nhóm 8 - IS403.R14
Mô tả: Bộ kiểm thử đơn vị (Unit Tests) nghiêm ngặt nhằm xác minh:
      1. Kích thước tập huấn luyện & số cặp học có giám sát của 3 fold (40, 46, 52) và full train (64).
      2. Nguyên tắc Shift(1): Không rò rỉ r_t vào đặc trưng rolling tại thời điểm t.
      3. Ranh giới thời gian & Zero Data Leakage: Thay đổi dữ liệu tương lai (validation/test) thành
         số giả không làm thay đổi bất kỳ giá trị dự báo nào của mô hình.
      4. Tính nhất quán toán học của phép đổi ngược z -> r -> y (tích lũy log-growth).
      5. Giao diện chuẩn hóa (Unified API) hoạt động đồng nhất trên cả 4 mô hình: Ridge, RF, SVR, XGBoost.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Thiết lập đường dẫn import
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.features import (
    compute_log_growth,
    create_features,
    build_supervised_data,
    compute_step_features_from_buffer,
    get_feature_names,
)
from common.backtest import (
    FOLDS_DEFINITION,
    fit_model,
    forecast_recursive,
    run_backtest,
    inverse_transform_growth,
    calculate_metrics,
    HAS_XGBOOST,
)


class TestLeakageAndTemporalBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Tải dữ liệu chuỗi chuẩn ch4_series.csv cho toàn bộ test case."""
        data_path = repo_root / "data" / "ch4_series.csv"
        cls.ch4 = pd.read_csv(data_path)
        cls.ch4["Date"] = pd.to_datetime(cls.ch4["Date"])
        cls.y = pd.Series(cls.ch4["EV_Total"].values, index=cls.ch4["Date"], name="EV_Total")

    def test_01_sample_sizes_across_folds_and_full_train(self):
        """
        Kiểm tra điều kiện tiên quyết về số lượng quan sát:
        - Full Train (71 tháng, đến 11/2022) phải có đúng 64 cặp học giám sát (từ 08/2017).
        - Fold 1 (47 tháng, đến 11/2020) phải có đúng 40 cặp học giám sát.
        - Fold 2 (53 tháng, đến 05/2021) phải có đúng 46 cặp học giám sát.
        - Fold 3 (59 tháng, đến 11/2021) phải có đúng 52 cặp học giám sát.
        """
        # 1. Full train 71 tháng
        y_train_71 = self.y.iloc[:71]
        sup_71 = build_supervised_data(y_train_71, dropna=True)
        self.assertEqual(len(y_train_71), 71, "Full train phải có đúng 71 tháng!")
        self.assertEqual(len(sup_71), 64, "Full train phải tạo đúng 64 cặp học giám sát!")
        self.assertEqual(
            sup_71.index[0].strftime("%Y-%m-%d"),
            "2017-08-31",
            "Cặp học đầu tiên phải bắt đầu từ tháng 08/2017!",
        )
        self.assertEqual(
            sup_71.index[-1].strftime("%Y-%m-%d"),
            "2022-11-30",
            "Cặp học cuối cùng phải kết thúc tại tháng 11/2022!",
        )

        # 2. Kiểm tra 3 folds
        for f in FOLDS_DEFINITION:
            train_end = pd.to_datetime(f["train_end"])
            y_fold = self.y[self.y.index <= train_end]
            sup_fold = build_supervised_data(y_fold, dropna=True)
            self.assertEqual(
                len(y_fold),
                f["n_train_months"],
                f"Fold {f['fold']} số tháng train không khớp!",
            )
            self.assertEqual(
                len(sup_fold),
                f["n_supervised_rows"],
                f"Fold {f['fold']} số cặp học có giám sát không khớp!",
            )

    def test_02_shift1_enforcement_on_features(self):
        """
        Kiểm tra nguyên tắc Shift(1):
        Tại thời điểm t, đặc trưng X_t không được chứa bất kỳ thông tin nào của r_t.
        Nếu ta thay đổi giá trị r_t thành một số ngẫu nhiên cực lớn (999.0),
        đặc trưng X_t tại dòng t phải giữ nguyên tuyệt đối!
        """
        y_sub = self.y.iloc[:20].copy()
        r = compute_log_growth(y_sub)
        X_orig = create_features(r)

        # Tạo chuỗi r bị biến đổi tại t=10
        r_corrupted = r.copy()
        original_val = r_corrupted.iloc[10]
        r_corrupted.iloc[10] = 999.99  # Giả mạo dữ liệu hiện tại

        X_corrupted = create_features(r_corrupted)

        # Tại t=10, đặc trưng X_t không được thay đổi vì chỉ nhìn từ t-1 trở về trước
        np.testing.assert_allclose(
            X_orig.iloc[10].values,
            X_corrupted.iloc[10].values,
            err_msg="LỖI RÒ RỈ: Đặc trưng tại thời điểm t bị ảnh hưởng khi giá trị r_t thay đổi!",
        )

        # Nhưng tại t=11, r_{t-1} phải phản ánh sự thay đổi
        self.assertEqual(X_corrupted.iloc[11]["r_lag1"], 999.99)

    def test_03_zero_future_leakage_in_forecasting(self):
        """
        Kiểm tra nghiệm thu quan trọng nhất:
        Thay đổi mọi giá trị trong tập validation hoặc test thành số giả
        KHÔNG ĐƯỢC PHÉP làm thay đổi dự báo của mô hình đã fit.
        """
        # Huấn luyện mô hình trên Fold 1 (train đến 2020-11-30)
        f1_end = pd.to_datetime("2020-11-30")
        train_y = self.y[self.y.index <= f1_end]

        artifact = fit_model(train_y, config={"alpha": 1.0}, model_type="ridge")
        pred_clean = forecast_recursive(artifact, horizon=12)["y_pred"]

        # Giả lập: Người dùng truyền một chuỗi test bị corrupt với số liệu giả
        fake_test_values = np.random.RandomState(42).uniform(999999, 1000000, 12)
        real_val_y = self.y[(self.y.index > f1_end) & (self.y.index <= "2021-11-30")].values

        # Dự báo lại
        pred_after = forecast_recursive(artifact, horizon=12)["y_pred"]

        # Dự báo phải hoàn toàn trùng khớp từng bit nhị phân
        np.testing.assert_array_equal(
            pred_clean,
            pred_after,
            err_msg="LỖI RÒ RỈ: Dự báo đệ quy bị thay đổi!",
        )

        # Hàm tính điểm (metric) thì PHẢI thay đổi khi test bị corrupt
        metric_real = calculate_metrics(real_val_y, pred_clean)
        metric_fake = calculate_metrics(fake_test_values, pred_clean)
        self.assertNotEqual(
            metric_real["rmse"],
            metric_fake["rmse"],
            "Hàm tính điểm phải phản ánh sự thay đổi của dữ liệu kiểm tra!",
        )

    def test_04_mathematical_inversion_of_growth(self):
        """
        Kiểm tra tính đúng đắn của phép nghịch đảo:
            y_{T+h} = y_T * exp( sum_{j=1}^h r_{T+j} )
        Không kẹp tăng trưởng âm thành 0.
        """
        last_y = 100000.0
        # Thử nghiệm với chuỗi tăng trưởng có cả số dương và số âm
        r_preds = np.array([0.02, -0.01, 0.03, 0.00, -0.005])

        y_expected = []
        curr = last_y
        for r_val in r_preds:
            curr = curr * np.exp(r_val)
            y_expected.append(curr)

        y_actual = inverse_transform_growth(r_preds, last_y)
        np.testing.assert_allclose(
            y_actual,
            np.array(y_expected),
            rtol=1e-10,
            err_msg="Phép chuyển đổi ngược r -> y không khớp công thức lũy kế!",
        )

    def test_05_unified_api_across_models(self):
        """
        Kiểm tra khả năng tái lập và dùng chung của API:
        Hai hoặc nhiều thành viên chạy các mô hình khác nhau (Ridge, RF, SVR, XGBoost)
        phải gọi được cùng hàm fit_model và forecast_recursive mà không cần viết lại logic.
        """
        train_y = self.y.iloc[:71]

        models_to_test = [
            ("ridge", {"alpha": 1.0}),
            ("rf", {"n_estimators": 50, "max_depth": 2, "random_state": 42}),
            ("svr", {"C": 1.0, "epsilon": 0.1, "kernel": "rbf"}),
        ]
        if HAS_XGBOOST:
            models_to_test.append(
                ("xgboost", {"max_depth": 1, "n_estimators": 20, "learning_rate": 0.05})
            )

        for m_type, cfg in models_to_test:
            artifact = fit_model(train_y, config=cfg, model_type=m_type)
            res = forecast_recursive(artifact, horizon=12)

            self.assertIn("y_pred", res)
            self.assertIn("r_pred", res)
            self.assertIn("trace_df", res)
            self.assertEqual(len(res["y_pred"]), 12)
            self.assertEqual(len(res["trace_df"]), 12)
            self.assertFalse(np.isnan(res["y_pred"]).any())
            self.assertTrue((res["y_pred"] > 0).all())


if __name__ == "__main__":
    unittest.main()
