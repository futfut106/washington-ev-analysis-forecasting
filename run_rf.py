"""Run: python run_rf.py --stage all (or prepare, cv, forecast, figures)."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import platform
import time
import warnings
import numpy as np
import pandas as pd
import sklearn
from common.features import build_supervised_data, create_features, compute_log_growth, get_feature_names
from common.backtest import calculate_metrics, FOLDS_DEFINITION, inverse_transform_growth
from models.random_forest import candidate_configs, fit, forecast, select_config

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIG = ROOT / "figures"
TRANSFORM = "standardized_log_growth"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def save_csv(df, name):
    df.to_csv(RES / name, index=False, encoding="utf-8-sig")


def load_inputs():
    df = pd.read_csv(ROOT / "data/ch4_series.csv", parse_dates=["date"])
    expected = pd.date_range("2017-01-31", "2023-11-30", freq="ME")
    assert df.date.equals(pd.Series(expected, name="date")), "Dates must be ordered and complete."
    assert len(df) == 83 and not df.date.duplicated().any()
    assert np.isfinite(df.ev_total).all() and (df.ev_total > 0).all()
    assert (df.ev_total == df.bev + df.phev).all()
    assert (df.total_vehicles == df.ev_total + df.non_ev).all()
    assert df.split.tolist() == ["train"] * 71 + ["test"] * 12
    y = df.set_index("date").ev_total.astype(float)
    protocol = json.loads((ROOT / "protocol_original.json").read_text(encoding="utf-8-sig"))
    folds = protocol["validation_protocol"]["folds"]
    assert folds == FOLDS_DEFINITION, "Protocol differs from shared framework."
    fc = pd.read_csv(ROOT / "data/folds.csv")
    pd.testing.assert_frame_equal(fc, pd.DataFrame(folds)[fc.columns], check_dtype=False)
    assert protocol["features"]["names"] == get_feature_names()
    assert len(build_supervised_data(y.iloc[:71])) == 64
    for f in folds:
        train = y.loc[:f["train_end"]]
        val = y.loc[f["val_start"]:f["val_end"]]
        assert len(train) == f["n_train_months"]
        assert len(build_supervised_data(train)) == f["n_supervised_rows"]
        assert len(val) == 12
    return df, y, protocol, folds


def prepare():
    RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    df, y, original, folds = load_inputs()
    actual_hash = digest(ROOT / "data/ch4_series.csv")
    spec = {
        "recorded_at_utc": now(), "data_sha256": actual_hash,
        "original_protocol_sha256": original["data"]["sha256"],
        "original_hash_matches": actual_hash == original["data"]["sha256"],
        "data_scope": "83 months supplied by user; train 71, comparison 12",
        "folds": folds, "feature_names": get_feature_names(),
        "target_transform": TRANSFORM, "target_std_ddof": 0,
        "configs": candidate_configs(), "primary_seed": 42,
        "selection": {"primary": "mean of three 12-step vehicle-scale fold RMSEs",
            "eligible": "mean_cv_rmse <= 1.01 * minimum mean_cv_rmse",
            "order": ["max_depth ascending", "min_samples_leaf descending",
                      "mean_cv_rmse ascending", "config_id ascending"],
            "note": "RF-specific clarification fixed before CV; max_features is not ranked by complexity"},
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in [
            ROOT/"common/features.py", ROOT/"common/backtest.py", ROOT/"models/random_forest.py", ROOT/"run_rf.py"]},
        "versions": {"python":platform.python_version(),"numpy":np.__version__,
                     "pandas":pd.__version__,"scikit_learn":sklearn.__version__},
    }
    write_json(ROOT / "protocol_rf.json", spec)
    save_csv(build_supervised_data(y.iloc[:71]).reset_index(), "supervised_train_rf.csv")
    rows = [dict(label=f"Fold {f['fold']}", **f) for f in folds]
    rows.append(dict(label="Full train",origin="2022-11",train_end="2022-11-30",
                     val_start="2022-12-31",val_end="2023-11-30",n_train_months=71,n_supervised_rows=64,val_months=12))
    save_csv(pd.DataFrame(rows), "data_checks_rf.csv")
    print("DATA CHECK: 83 months | train=71 | test=12 | full supervised=64")
    print(pd.DataFrame(rows)[["label","train_end","n_train_months","n_supervised_rows"]].to_string(index=False))
    print("DATA HASH MATCHES ORIGINAL PROTOCOL:",spec["original_hash_matches"])
    print("RF-specific selection order saved BEFORE CV in protocol_rf.json")
    return spec


def check_protocol():
    p = json.loads((ROOT / "protocol_rf.json").read_text(encoding="utf-8"))
    assert p["data_sha256"] == digest(ROOT / "data/ch4_series.csv"), "Input changed since prepare."
    assert p["configs"] == candidate_configs(), "Grid changed since prepare."
    for name, value in p["source_sha256"].items():
        assert digest(ROOT / name) == value, f"Code changed since prepare: {name}"
    return p


def run_cv():
    protocol = check_protocol()
    _, y, _, folds = load_inputs()
    # Only the first 71 months are exposed to the model-selection loop.
    selection_history = y.iloc[:71].copy()
    cv_rows, forecast_rows = [], []
    start = time.perf_counter()
    for cfg in candidate_configs():
        for f in folds:
            row = dict(config_id=cfg["config_id"],fold=f["fold"],origin=f["origin"],
                       n_train_months=f["n_train_months"],n_supervised_rows=f["n_supervised_rows"])
            observed = []
            try:
                train = selection_history.loc[:f["train_end"]]
                with warnings.catch_warnings(record=True) as observed:
                    warnings.simplefilter("always")
                    artifact = fit(train, cfg)
                    prediction = forecast(artifact, horizon=12)
                assert np.isfinite(prediction["y_pred"]).all()
                assert artifact["n_supervised_rows"] == f["n_supervised_rows"]
                validation = selection_history.loc[f["val_start"]:f["val_end"]]
                row.update(calculate_metrics(validation.values, prediction["y_pred"]),status="SUCCESS")
                for i, (date, actual) in enumerate(validation.items()):
                    forecast_rows.append(dict(config_id=cfg["config_id"],fold=f["fold"],
                        date=str(date.date()), origin=f["train_end"],horizon=i+1,actual=actual,pred=float(prediction["y_pred"][i])))
            except Exception as e:
                row.update(rmse=np.nan,mae=np.nan,smape=np.nan,bias=np.nan,status="FAILED",error=repr(e))
            row["warnings"] = " | ".join(sorted(set(str(w.message) for w in observed))) or "None"
            cv_rows.append(row)
            print(f"{cfg['config_id']} fold={f['fold']} status={row['status']} RMSE={row['rmse']:.4f}",flush=True)
    cv = pd.DataFrame(cv_rows)
    save_csv(cv,"cv_rf.csv")
    save_csv(pd.DataFrame(forecast_rows),"cv_forecasts_rf.csv")
    if not (cv.status == "SUCCESS").all():
        raise RuntimeError("CV contains failures; inspect cv_rf.csv. No config selected and no test evaluated.")
    summary = pd.DataFrame(candidate_configs()).merge(
        cv.groupby("config_id").agg(mean_cv_rmse=("rmse","mean"),mean_cv_mae=("mae","mean"),
            mean_cv_smape=("smape","mean"),mean_cv_bias=("bias","mean")).reset_index(),on="config_id")
    for f in (1,2,3):
        summary[f"rmse_fold{f}"] = summary.config_id.map(cv[cv.fold==f].set_index("config_id").rmse)
    identifier, minimum, eligible = select_config(summary)
    summary["within_1pct"] = summary.config_id.isin(eligible)
    summary["selected"] = summary.config_id == identifier
    summary = summary.sort_values(["mean_cv_rmse","config_id"])
    save_csv(summary,"cv_rf_summary.csv")
    selected = next(c for c in candidate_configs() if c["config_id"]==identifier)
    lock = dict(config=selected, locked_at_utc=now(), data_sha256=protocol["data_sha256"],
                protocol_sha256=digest(ROOT/"protocol_rf.json"), mean_cv_rmse=float(
                    summary.loc[summary.selected,"mean_cv_rmse"].iloc[0]),
                minimum_mean_cv_rmse=minimum,eligible_config_ids=eligible,
                selection_rule=protocol["selection"],cv_runtime_s=time.perf_counter()-start,
                versions=protocol["versions"],status="LOCKED_BEFORE_TEST")
    write_json(RES/"best_rf.json",lock)
    print("\nVALIDATION SUMMARY\n",summary[["config_id","max_depth","min_samples_leaf","max_features",
        "rmse_fold1","rmse_fold2","rmse_fold3","mean_cv_rmse","selected"]].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("LOCKED:",identifier,"| best_rf.json saved before test")
    return summary


def run_forecast():
    protocol = check_protocol()
    lock = json.loads((RES/"best_rf.json").read_text(encoding="utf-8"))
    assert lock["protocol_sha256"] == digest(ROOT/"protocol_rf.json"), "Protocol changed since config locked."
    assert lock["data_sha256"] == protocol["data_sha256"]
    _, y, _, _ = load_inputs()
    train = y.iloc[:71].copy()
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        artifact=fit(train,lock["config"])
        fc=forecast(artifact,horizon=12)
    assert np.isfinite(fc["y_pred"]).all() and len(fc["y_pred"])==12
    assert artifact["scaler_X"] is None
    test=y.iloc[71:]
    result=pd.DataFrame(dict(model="Random Forest",date=test.index.strftime("%Y-%m-%d"),
        origin="2022-11-30",horizon=np.arange(1,13),actual=test.values,pred=fc["y_pred"],
        config_id=lock["config"]["config_id"],target_transform=TRANSFORM))
    save_csv(result,"forecast_rf.csv")
    trace=fc["trace_df"].copy();trace.insert(0,"date",result.date.values)
    trace["actual"]=test.values;trace["error_actual_minus_pred"]=test.values-fc["y_pred"]
    save_csv(trace,"trace_rf.csv")
    metrics=calculate_metrics(test.values,fc["y_pred"])
    metrics.update(model="Random Forest",config_id=lock["config"]["config_id"],mean_cv_rmse=lock["mean_cv_rmse"],
        n_train=71,n_supervised=64,seed=42,runtime_s=time.perf_counter()-start)
    save_csv(pd.DataFrame([metrics]),"metrics_rf.csv")
    # Focused checks of this RF integration. No original test suite was provided.
    checks=[]
    def check(name,condition):
        checks.append(dict(check=name,status="PASS" if condition else "FAIL"))
        if not condition: raise AssertionError(name)
    check("full_train_64_rows",artifact["n_supervised_rows"]==64)
    sup=build_supervised_data(train)
    check("label_scaler_train_only",np.isclose(artifact["mu_r"],sup.r.mean()) and
          np.isclose(artifact["std_r"],sup.r.std(ddof=0)))
    check("recursive_lag1_uses_previous_prediction",np.allclose(trace.r_lag1.values[1:],trace.r_pred.values[:-1],rtol=0,atol=0))
    r=compute_log_growth(train);changed=r.copy();changed.iloc[20]=999
    check("features_exclude_current_target",np.array_equal(create_features(r).iloc[20].values,create_features(changed).iloc[20].values))
    altered=y.copy();altered.iloc[71:]=np.arange(12)*10000+10
    altered_fc=forecast(artifact,train_history=altered.loc[:"2022-11-30"],horizon=12)
    baseline=forecast(artifact,train_history=train,horizon=12)
    check("future_actual_mutation_does_not_change_predictions",np.array_equal(baseline["y_pred"],altered_fc["y_pred"]))
    # Each validation forecast API is also tested after fit with a changed future tail.
    cvs=pd.read_csv(RES/"cv_forecasts_rf.csv")
    check("cv_288_predictions",len(cvs)==8*3*12)
    expected=train.iloc[-1]*np.exp(np.cumsum(trace.r_pred))
    check("growth_inversion",np.allclose(expected,fc["y_pred"],rtol=1e-12))
    check("negative_growth_not_clipped",inverse_transform_growth([-0.1],100)[0]<100)
    save_csv(pd.DataFrame(checks),"integration_checks_rf.csv")
    metadata=dict(mu_r=artifact["mu_r"],std_r=artifact["std_r"],std_ddof=0,
        last_y=artifact["last_y"],n_train_months=71,n_supervised_rows=64,
        feature_names=artifact["feature_names"],scaler_X=None,
        estimator_params=artifact["model"].get_params(),
        warnings=[str(w.message) for w in observed],
        forecast_generated_at_utc=now(),locked_at_utc=lock["locked_at_utc"],
        source_data_sha256=protocol["data_sha256"])
    write_json(RES/"fit_metadata_rf.json",metadata)
    arima=pd.read_csv(RES/"forecast_arima.csv")
    assert arima.date.tolist()==result.date.tolist()
    assert arima.origin.eq("2022-11-30").all() and arima.horizon.tolist()==list(range(1,13))
    assert np.array_equal(arima.actual.values,test.values)
    reference=calculate_metrics(arima.actual.values,arima.pred.values)
    comparison=pd.DataFrame([dict(model="ARIMA(2,2,2) supplied forecast",**reference),dict(model="Random Forest",**{k:metrics[k] for k in reference})])
    comparison["rmse_improvement_vs_arima_pct"]=100*(reference["rmse"]-comparison.rmse)/reference["rmse"]
    save_csv(comparison,"comparison_rf_arima.csv")
    print("\n12-MONTH FORECAST\n",result[["date","horizon","actual","pred"]].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\nRF TEST METRICS\n",pd.DataFrame([metrics]).to_string(index=False))
    print("\nREFERENCE RECOMPUTED FROM SUPPLIED CSV\n",comparison.to_string(index=False))
    print("\nINTEGRATION CHECKS\n",pd.DataFrame(checks).to_string(index=False))
    return result


def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import StrMethodFormatter
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    cv=pd.read_csv(RES/"cv_rf_summary.csv").sort_values("config_id")
    result=pd.read_csv(RES/"forecast_rf.csv",parse_dates=["date"])
    fig,ax=plt.subplots(figsize=(8.8,3.7))
    for k in (1,2,3):ax.plot(cv.config_id,cv[f"rmse_fold{k}"],marker="o",label=f"Fold {k}",alpha=.7)
    ax.plot(cv.config_id,cv.mean_cv_rmse,color="#162b49",linewidth=2.5,marker="s",label="Trung bình 3 fold")
    ax.set(ylabel="RMSE (xe)",xlabel="Cấu hình",title="Random Forest trên ba cửa sổ validation")
    ax.grid(axis="y",alpha=.2);ax.legend(ncol=4,fontsize=8);fig.tight_layout()
    fig.savefig(FIG/"rf_validation.png",dpi=240);plt.close(fig)
    _,y,_,_=load_inputs()
    fig,ax=plt.subplots(figsize=(8.8,3.5))
    hist=y.loc["2022-07-31":"2022-11-30"]
    ax.plot(hist.index,hist.values,color="#727a85",label="Lịch sử cuối train")
    ax.plot(result.date,result.actual,color="#143d66",marker="o",label="Thực tế")
    dates=pd.DatetimeIndex([pd.Timestamp("2022-11-30"),*result.date])
    ax.plot(dates,[y.iloc[70],*result.pred],color="#c05a25",marker="s",markersize=4,label="RF dự báo đệ quy")
    ax.axvline(pd.Timestamp("2022-11-30"),color="#555",linestyle="--",linewidth=.9)
    ax.set(ylabel="EV Total (xe)",title="Dự báo Random Forest từ gốc 11/2022 cho 12 tháng")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2));ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.grid(axis="y",alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(FIG/"rf_forecast.png",dpi=260);plt.close(fig)
    err=result.actual-result.pred
    fig,ax=plt.subplots(figsize=(8.8,3))
    ax.bar(result.horizon,err,color=np.where(err>=0,"#c05a25","#167777"))
    ax.axhline(0,color="black",linewidth=.8)
    ax.set(xlabel="Chân trời h (tháng)",ylabel="Thực tế − dự báo (xe)",title="Sai số có dấu của Random Forest",xticks=list(range(1,13)))
    ax.grid(axis="y",alpha=.2);fig.tight_layout();fig.savefig(FIG/"rf_errors.png",dpi=240);plt.close(fig)
    # Render exact output tables as reusable screenshots, with only display values rounded.
    display=cv[["config_id","max_depth","min_samples_leaf","max_features","rmse_fold1","rmse_fold2","rmse_fold3","mean_cv_rmse"]].copy()
    for c in display.columns[4:]:display[c]=display[c].map(lambda x:f"{x:,.2f}")
    fig,ax=plt.subplots(figsize=(11.2,3.4));ax.axis("off")
    tab=ax.table(cellText=display.values,colLabels=["Cấu hình","Độ sâu","Min lá","Max features","RMSE F1","RMSE F2","RMSE F3","Mean RMSE"],loc="center",cellLoc="center")
    tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,1.5)
    for j in range(8):tab[(0,j)].set_facecolor("#dfe8f2")
    for i,selected in enumerate(cv.selected,1):
        if selected:
            for j in range(8):tab[(i,j)].set_facecolor("#e5f2e7")
    ax.set_title("Validation đủ 8 cấu hình RF — dòng xanh là cấu hình được chọn",pad=12)
    fig.tight_layout();fig.savefig(FIG/"rf_validation_table.png",dpi=240,bbox_inches="tight");plt.close(fig)
    print("Figures saved in figures/")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["all","prepare","cv","forecast","figures"],default="all")
    args=parser.parse_args()
    if args.stage in ("all","prepare"):prepare()
    if args.stage in ("all","cv"):run_cv()
    if args.stage in ("all","forecast"):run_forecast()
    if args.stage in ("all","figures"):make_figures()


if __name__=="__main__":main()
