"""Ridge benchmark for monthly Washington EV registrations."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

ALPHAS = (0.1, 1.0, 10.0)
FOLD_SIZES = (47, 53, 59)
HORIZON = 12
FEATURES = ("r_lag_1", "r_lag_2", "r_lag_3", "r_mean_3", "r_mean_6")


def feature_row(history):
    return [history[-1], history[-2], history[-3],
            np.mean(history[-3:]), np.mean(history[-6:])]


def standardize(values):
    center = values.mean(axis=0)
    scale = np.where(values.std(axis=0) == 0, 1, values.std(axis=0))
    return (values - center) / scale, center, scale


def fit(train_history, config):
    """Fit using the history available at a single origin."""
    levels = np.asarray(train_history, dtype=float)
    if len(levels) < 8 or not np.isfinite(levels).all() or np.any(levels <= 0):
        raise ValueError("Expected at least eight finite positive monthly levels.")
    growth = np.diff(np.log(levels))
    features = np.array([feature_row(growth[:i]) for i in range(6, len(growth))])
    x, x_mean, x_scale = standardize(features)
    z, r_mean, r_scale = standardize(growth[6:])
    alpha = float(config["alpha"])
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and positive.")
    # Equivalent to StandardScaler followed by Ridge with an unpenalized intercept.
    weights = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (z - z.mean()))
    return dict(weights=weights, intercept=z.mean(), x_mean=x_mean, x_scale=x_scale,
                r_mean=r_mean, r_scale=r_scale, config={"alpha": alpha},
                n_supervised=len(z))


def forecast(artifact, train_history, horizon=12):
    """Return vehicle counts and a trace without accepting future observations."""
    levels = np.asarray(train_history, dtype=float)
    growth = np.diff(np.log(levels)).tolist()
    level = float(levels[-1])
    trace = []
    for step in range(1, horizon + 1):
        features = np.asarray(feature_row(growth))
        x = (features - artifact["x_mean"]) / artifact["x_scale"]
        z = artifact["intercept"] + x @ artifact["weights"]
        rate = float(artifact["r_mean"] + artifact["r_scale"] * z)
        level *= float(np.exp(rate))
        if not np.isfinite(level):
            raise ValueError("Non-finite forecast.")
        growth.append(rate)
        trace.append(dict(horizon=step, pred=level, pred_log_growth=rate,
                          **dict(zip(FEATURES, features))))
    return trace


def metrics(actual, pred):
    actual, pred = np.asarray(actual), np.asarray(pred)
    error = actual - pred
    denom = abs(actual) + abs(pred)
    relative = np.divide(2 * abs(error), denom, out=np.zeros_like(error, dtype=float),
                         where=denom != 0)
    return dict(rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(abs(error))),
                smape=float(100 * relative.mean()), bias=float(error.mean()))


def run(source, output):
    frame = pd.read_csv(source, parse_dates=["date"])
    dates = pd.date_range("2017-01-31", periods=83, freq="ME")
    if (not pd.DatetimeIndex(frame.date).equals(dates)
            or frame.split.tolist() != ["train"] * 71 + ["test"] * 12
            or not (frame.bev + frame.phev).eq(frame.ev_total).all()
            or not (frame.ev_total + frame.non_ev).eq(frame.total_vehicles).all()):
        raise ValueError("Expected the verified 83-month Chapter 4 input.")
    output.mkdir(parents=True, exist_ok=True)
    values = frame.ev_total.to_numpy()
    train = values[:71]
    cv = []
    scores = {}
    for alpha in ALPHAS:
        config_id = f"ridge_alpha_{alpha:g}"
        for fold, size in enumerate(FOLD_SIZES, 1):
            artifact = fit(train[:size], {"alpha": alpha})
            trace = forecast(artifact, train[:size], HORIZON)
            score = metrics(train[size:size+HORIZON], [row["pred"] for row in trace])
            cv.append(dict(config_id=config_id, fold=fold, origin=dates[size-1].date(),
                           n_train_months=size, n_supervised_rows=artifact["n_supervised"],
                           **score, status="ok", warnings=""))
        scores[alpha] = float(np.mean([row["rmse"] for row in cv[-3:]]))
    pd.DataFrame(cv).to_csv(output / "cv_ridge.csv", index=False)
    threshold = min(scores.values()) * 1.01
    selected = max(alpha for alpha in ALPHAS if scores[alpha] <= threshold)
    config_id = f"ridge_alpha_{selected:g}"
    artifact = fit(train, {"alpha": selected})
    config = dict(config_id=config_id, alpha=selected, cv_mean_rmse=scores,
                  tie_rule="Largest alpha within 1% of minimum mean CV RMSE",
                  n_supervised=artifact["n_supervised"], fold_sizes=FOLD_SIZES,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)
    (output / "best_ridge.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    trace = forecast(artifact, train, HORIZON)
    predictions = [row["pred"] for row in trace]
    rows = [dict(model="Ridge", date=dates[71+i].date(), origin=dates[70].date(),
                 horizon=i+1, actual=int(values[71+i]), pred=pred,
                 config_id=config_id, target_transform="standardized_log_growth")
            for i, pred in enumerate(predictions)]
    pd.DataFrame(rows).to_csv(output / "forecast_ridge.csv", index=False)
    pd.DataFrame(trace).to_csv(output / "trace_ridge.csv", index=False)
    score = metrics(values[71:], predictions)
    (output / "ridge_test_metrics.json").write_text(json.dumps(score, indent=2), encoding="utf-8")
    return config, score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/ch4_series.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    config, score = run(args.input, args.output)
    print(json.dumps(dict(config=config, test=score), indent=2))
