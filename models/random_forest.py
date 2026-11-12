"""RF grid and wrappers using TV2's original shared framework."""
from itertools import product
from common.backtest import fit_model, forecast_recursive


def candidate_configs():
    return [dict(config_id=f"RF{i:02d}", n_estimators=300, max_depth=d,
                 min_samples_leaf=l, max_features=f, random_state=42)
            for i, (d, l, f) in enumerate(product((2, 3), (3, 5), (0.7, 1.0)), 1)]


def fit(train_history, config):
    return fit_model(train_history, config=config, model_type="rf", needs_scaling=False)


def forecast(artifact, train_history=None, horizon=12):
    # No future actuals are accepted or read by this wrapper.
    return forecast_recursive(artifact, train_history=train_history, horizon=horizon)


def select_config(summary):
    best_rmse = float(summary["mean_cv_rmse"].min())
    eligible = summary.loc[summary.mean_cv_rmse <= best_rmse * 1.01].copy()
    # Recorded before CV in protocol_rf.json. max_features is not ranked
    # as a complexity measure; break equal depth/leaf by validation RMSE.
    chosen = eligible.sort_values(
        ["max_depth", "min_samples_leaf", "mean_cv_rmse", "config_id"],
        ascending=[True, False, True, True], kind="stable").iloc[0]
    return str(chosen.config_id), best_rmse, eligible.config_id.tolist()
