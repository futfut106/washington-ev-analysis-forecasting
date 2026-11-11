"""Phân tích mục 4.2 của bộ dữ liệu History Washington 01/2017–11/2023.

Cách chạy:
    python ch4_2_analysis.py
    python ch4_2_analysis.py --input "duong_dan/file.csv" --output "ket_qua"

Không cần mở Word hoặc kết nối Google Drive. Không ghi đè dữ liệu đầu vào.
Toàn chuỗi dùng để mô tả; phần train chỉ được xuất để đối chiếu, không chọn mô hình.
"""

import argparse
import json
import hashlib
import importlib.metadata
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Lưu hình được cả trên máy không có giao diện đồ họa.
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

COUNT_COLUMNS = [
    "Battery Electric Vehicles (BEVs)",
    "Plug-In Hybrid Electric Vehicles (PHEVs)",
    "Electric Vehicle (EV) Total",
    "Non-Electric Vehicle Total",
    "Total Vehicles",
]


def require(condition, message):
    """Kiểm tra vẫn hoạt động khi Python chạy với tùy chọn -O."""
    if not condition:
        raise ValueError(message)


def load_series(input_path):
    """Kiểm tra đúng bản dữ liệu giao trong task trước khi tổng hợp."""
    h = pd.read_csv(input_path, encoding="utf-8-sig")
    required = ["Date", "County", "State", "Vehicle Primary Use", *COUNT_COLUMNS,
                "Percent Electric Vehicles"]
    missing = sorted(set(required) - set(h.columns))
    require(not missing, f"Thiếu cột bắt buộc: {missing}")
    require(len(h) == 6474, "Task này yêu cầu bản History gồm đúng 6.474 dòng.")
    require(not h[required].isna().any().any(), "Dữ liệu có ô thiếu.")
    h["Date"] = pd.to_datetime(h["Date"], errors="raise")
    h[COUNT_COLUMNS] = h[COUNT_COLUMNS].apply(pd.to_numeric, errors="raise")
    counts = h[COUNT_COLUMNS].to_numpy(dtype=float)
    require(np.isfinite(counts).all(), "Các cột số lượng có giá trị không hữu hạn.")
    require((counts >= 0).all() and (counts == np.floor(counts)).all(),
            "Số lượng xe phải là số nguyên không âm.")
    require(h["State"].eq("WA").all(), "Dữ liệu phải chỉ gồm State = WA.")
    require(h["County"].nunique() == 39, "Dữ liệu phải có đúng 39 county.")
    require(set(h["Vehicle Primary Use"]) == {"Passenger", "Truck"},
            "Vehicle Primary Use phải gồm Passenger và Truck.")
    require(h["Date"].dt.is_month_end.all(), "Date phải là ngày cuối tháng.")
    require(not h.duplicated(["Date", "County", "Vehicle Primary Use"]).any(),
            "Trùng khóa Date × County × Vehicle Primary Use.")
    require(h.groupby("Date").size().eq(78).all(), "Mỗi tháng phải đủ 78 tổ hợp.")
    require(h[COUNT_COLUMNS[2]].eq(h[COUNT_COLUMNS[0]] + h[COUNT_COLUMNS[1]]).all(),
            "Có dòng không thỏa EV Total = BEV + PHEV.")
    require(h[COUNT_COLUMNS[4]].eq(h[COUNT_COLUMNS[2]] + h[COUNT_COLUMNS[3]]).all(),
            "Có dòng không thỏa Total Vehicles = EV + Non-EV.")
    s = h.groupby("Date")[COUNT_COLUMNS].sum().sort_index()
    require(len(s) == 83 and s.index.to_period("M").equals(
        pd.period_range("2017-01", "2023-11", freq="M")),
        "Cần đúng 83 tháng liên tục từ 01/2017 đến 11/2023.")
    s.columns = ["bev", "phev", "ev_total", "non_ev", "total_vehicles"]
    require(s.ev_total.gt(0).all() and s.total_vehicles.gt(0).all(),
            "Chuỗi tổng phải dương để tính tỷ lệ và log.")
    # Tính lại tỷ lệ từ tổng số lượng, không lấy trung bình tỷ lệ của county.
    s["bev_share"] = 100 * s.bev / s.ev_total
    s["ev_pct"] = 100 * s.ev_total / s.total_vehicles
    s["delta"] = s.ev_total.diff()
    s["mom"] = s.ev_total.pct_change(fill_method=None) * 100
    s["yoy_delta"] = s.ev_total.diff(12)
    s["yoy"] = s.ev_total.pct_change(12, fill_method=None) * 100
    s["ma12"] = s.ev_total.rolling(12, min_periods=12).mean()
    s["sd12"] = s.delta.rolling(12, min_periods=12).std(ddof=1)
    s["log_growth"] = 100 * np.log(s.ev_total).diff()
    s["split"] = np.where(s.index <= pd.Timestamp("2022-11-30"), "train", "test")
    s.attrs.update(source_rows=len(h), counties=h.County.nunique(), primary_uses=h['Vehicle Primary Use'].nunique())
    return s


def summarize(s):
    """Số liệu thô chưa làm tròn; định dạng tiếng Việt thực hiện ở báo cáo."""
    d = s.delta.dropna()
    lg = s.log_growth.dropna()
    summary = {}
    intervals = len(s) - 1  # 83 mốc tương ứng 82 khoảng tháng.
    for key in ["bev", "phev", "ev_total"]:
        a, b = s[key].iloc[[0, -1]]
        summary[key] = dict(start=a, end=b, change=b-a,
                            growth_pct=100*(b/a-1),
                            cagr=100*((b/a)**(12/intervals)-1))
    half = len(d) // 2
    for name, z in [("first41", d.iloc[:half]), ("last41", d.iloc[half:]),
                    ("first24", d.iloc[:24]), ("last24", d.iloc[-24:])]:
        summary[name] = dict(start=str(z.index[0].date()), end=str(z.index[-1].date()),
                             mean=z.mean(), sd=z.std(ddof=1), n=len(z),
                             log_mean=lg.loc[z.index].mean(), log_sd=lg.loc[z.index].std(ddof=1))
    for name, z in [("level", s.ev_total), ("delta", d), ("log_growth", lg),
                    ("train_delta", s.loc[s.split.eq("train"), "ev_total"].diff().dropna())]:
        values = acf(z, nlags=24, fft=False, adjusted=False, missing="raise")
        summary["acf_" + name] = {str(i): values[i] for i in [1, 2, 3, 12, 24]}
    summary["share"] = [s.bev_share.iloc[0], s.bev_share.iloc[-1]]
    selected = d.nlargest(3).index.union(d.nsmallest(2).index)
    summary["extremes"] = (s.loc[selected, ["ev_total", "delta", "mom", "yoy"]]
        .rename_axis("Date").reset_index().astype({"Date": str}).to_dict("records"))
    summary["yoy_extremes"] = dict(min=s.yoy.min(), min_date=str(s.yoy.idxmin().date()),
                                   max=s.yoy.max(), max_date=str(s.yoy.idxmax().date()),
                                   end=s.yoy.iloc[-1])
    summary["ma12_end"] = s.ma12.iloc[-1]
    summary["negative_months"] = int((d < 0).sum())
    summary["bev_contribution"] = 100 * summary["bev"]["change"] / summary["ev_total"]["change"]
    summary["year_end"] = (s.loc[s.index.month == 12, ["ev_total", "yoy"]]
        .rename_axis("Date").reset_index().astype({"Date": str}).to_dict("records"))
    return clean_json(summary)


def clean_json(value):
    """Chuyển mọi NaN thành null; không sửa tay một dòng cụ thể."""
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def export_tables(s, summary, output_dir):
    s.to_csv(output_dir / "ch4_2_series.csv", index_label="date", encoding="utf-8-sig")
    (output_dir / "stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tables = {
        "notable_months.csv": pd.DataFrame(summary["extremes"]),
        "table4_1_growth.csv": pd.DataFrame({k: summary[k] for k in ["bev", "phev", "ev_total"]}).T.rename_axis("series"),
        "table4_2_variability.csv": pd.DataFrame({k: summary[k] for k in ["first41", "last41"]}).T.rename_axis("period"),
        "acf_values.csv": pd.DataFrame({k: summary["acf_"+k] for k in ["level", "delta", "log_growth", "train_delta"]}).rename_axis("lag"),
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=filename != "notable_months.csv", encoding="utf-8-sig")
    # Các bảng bổ trợ giúp kiểm tra đủ thay đổi tuyệt đối và phần trăm theo năm.
    comparison = pd.DataFrame(index=s.index)
    for name in ["bev", "phev", "ev_total"]:
        comparison[name] = s[name]
        comparison[name + "_mom_pct"] = s[name].pct_change(fill_method=None) * 100
        comparison[name + "_yoy_change"] = s[name].diff(12)
        comparison[name + "_yoy_pct"] = s[name].pct_change(12, fill_method=None) * 100
    comparison.to_csv(output_dir / "component_growth.csv", index_label="date", encoding="utf-8-sig")
    comparison.loc[(s.index.month == 12) | (s.index == s.index[-1])].to_csv(
        output_dir / "annual_growth.csv", index_label="date", encoding="utf-8-sig")
    calendar = pd.DataFrame({"year": s.index.year, "month": s.index.month,
                             "mom": s["mom"].to_numpy()})
    calendar.pivot(index="year", columns="month", values="mom").to_csv(
        output_dir / "seasonal_growth_matrix.csv", encoding="utf-8-sig")


def make_figures(s, output_dir):
    """Tạo 5 hình của mục 4.2; dải ACF chỉ là tham chiếu nhiễu trắng."""
    d = s.delta.dropna()
    lg = s.log_growth.dropna()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.2, 'figure.dpi': 170})
    colors = ['#163f66', '#148583', '#c17626']

    def save(fig, n):
        for ax in fig.axes:
            if ax.get_ylabel() != 'Năm':
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'.replace('.', ',')))
        for ax in fig.axes:
            if ax.xaxis.get_converter() is not None:
                ax.set_xlim(s.index[0], s.index[-1])
                years = [pd.Timestamp(year=year, month=1, day=31)
                         for year in range(s.index[0].year, s.index[-1].year + 1)]
                ax.set_xticks(years, [str(date.year) for date in years])
        fig.tight_layout(pad=0.6)
        fig.savefig(output_dir / f'fig{n}.png', dpi=210, bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.0, 2.1))
    for k, label, c in zip(['ev_total', 'bev', 'phev'], ['EV Total', 'BEV', 'PHEV'], colors):
        ax.plot(s.index, s[k] / 1000, label=label, color=c, lw=1.8)
    ax.set_ylabel('Nghìn xe')
    ax.legend(ncol=3, loc='upper left', frameon=False)
    save(fig, 1)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 1.9))
    axs[0].stackplot(s.index, s.bev_share, 100 - s.bev_share,
                     labels=['BEV', 'PHEV'], colors=[colors[1], colors[2]], alpha=0.85)
    axs[0].set_ylim(0, 100)
    axs[0].set_ylabel('Cơ cấu EV (%)')
    axs[0].legend(fontsize=7, ncol=2, loc='lower right', frameon=False)
    for name, color in [('bev', colors[1]), ('phev', colors[2])]:
        axs[1].plot(s.index, 100 * s[name] / s[name].iloc[0],
                    label=name.upper(), color=color, lw=1.5)
    axs[1].set_ylabel('Chỉ số (01/2017 = 100)')
    axs[1].legend(fontsize=7, frameon=False)
    save(fig, 2)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.0))
    axs[0].plot(s.index, s.ev_total / 1000, color=colors[0], lw=1, label='EV Total')
    axs[0].plot(s.index, s.ma12 / 1000, color=colors[2], lw=1.5, label='TB trượt 12 tháng')
    axs[0].set_ylabel('Nghìn xe')
    axs[0].legend(fontsize=7, frameon=False)
    axs[1].plot(s.index, s.mom, label='So tháng trước', lw=1, color=colors[1])
    axs[1].plot(s.index, s.yoy, label='So cùng kỳ năm trước', lw=1.3, color=colors[0])
    axs[1].set_ylabel('Tăng trưởng (%)')
    axs[1].legend(fontsize=7, frameon=False)
    save(fig, 3)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.1))
    calendar = pd.DataFrame({'year': s.index.year, 'month': s.index.month,
                              'mom': s.mom.to_numpy()})
    matrix = calendar.pivot(index='year', columns='month', values='mom')
    # Tháng không có dữ liệu không được coi là 0%.
    cmap = plt.get_cmap('viridis').with_extremes(bad='#dddddd')
    heat = axs[0].imshow(np.ma.masked_invalid(matrix.to_numpy()), aspect='auto', cmap=cmap,
                          extent=(0.5, 12.5, len(matrix)-0.5, -0.5), vmin=-0.02, vmax=5)
    axs[0].grid(False)
    axs[0].set_yticks(range(len(matrix)), matrix.index)
    axs[0].set_xticks([1, 3, 6, 9, 12])
    axs[0].set_xlabel('Tháng trong năm')
    axs[0].set_ylabel('Năm')
    bar = fig.colorbar(heat, ax=axs[0], fraction=0.04, pad=0.025)
    bar.set_label('Tăng tháng (%)', fontsize=7)
    bar.ax.tick_params(labelsize=7)
    bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'.replace('.', ',')))
    for name, z, c in [('Mức gốc', s.ev_total, colors[0]), ('Chênh lệch tháng', d, colors[2]), ('Chênh lệch log', lg, colors[1])]:
        a = acf(z, nlags=24, fft=False)
        axs[1].plot(range(1, 25), a[1:], marker='.', ms=3, lw=1, label=name, color=c)
    bound = 1.96 / np.sqrt(len(lg))
    axs[1].axhline(bound, color='gray', ls='--', lw=0.7)
    axs[1].axhline(-bound, color='gray', ls='--', lw=0.7)
    axs[1].axhline(0, color='gray', lw=0.5)
    axs[1].set_xticks([1, 6, 12, 18, 24])
    axs[1].set_xlabel('Độ trễ (tháng)')
    axs[1].set_ylabel('ACF')
    axs[1].legend(fontsize=6.5, frameon=False)
    save(fig, 4)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 1.95))
    axs[0].bar(s.index, s.delta, width=22,
                color=np.where(s.delta.lt(0), '#b44747', colors[0]))
    axs[0].axhline(0, color='gray', lw=0.6)
    axs[0].set_ylabel('Thay đổi tháng (xe)')
    for dt in [d.idxmin(), d.idxmax()]:
        axs[0].scatter(dt, d[dt], color=colors[2], s=15)
        axs[0].annotate(dt.strftime('%m/%Y'), (dt, d[dt]), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=6)
    axs[1].plot(s.index, s.sd12, color=colors[1], lw=1.5)
    axs[1].set_ylabel('Độ lệch chuẩn trượt (xe)')
    save(fig, 5)


def export_report_metrics(s, summary, input_path, output_dir):
    """Nguồn duy nhất cho mọi số liệu được chèn vào bản Word."""
    m = dict(summary)
    m.update(rows=s.attrs['source_rows'], counties=s.attrs['counties'],
             primary_uses=s.attrs['primary_uses'], months=len(s), intervals=len(s)-1,
             changes=len(s.delta.dropna()), half_changes=len(s.delta.dropna())//2,
             start_date=str(s.index[0].date()), end_date=str(s.index[-1].date()),
             train_months=int(s.split.eq('train').sum()), test_months=int(s.split.eq('test').sum()),
             share_change=s.bev_share.iloc[-1]-s.bev_share.iloc[0],
             phev_share_start=100-s.bev_share.iloc[0], phev_share_end=100-s.bev_share.iloc[-1],
             share_min=s.bev_share.min(), share_min_date=str(s.bev_share.idxmin().date()),
             half_mean_ratio=summary['last41']['mean']/summary['first41']['mean'],
             half_sd_ratio=summary['last41']['sd']/summary['first41']['sd'],
             acf_bound=1.96/np.sqrt(len(s.log_growth.dropna())),
             ev_multiple=summary['ev_total']['end']/summary['ev_total']['start'],
             bev_multiple=summary['bev']['end']/summary['bev']['start'],
             phev_multiple=summary['phev']['end']/summary['phev']['start'],
             bev_index=100*summary['bev']['end']/summary['bev']['start'],
             phev_index=100*summary['phev']['end']/summary['phev']['start'],
             bev_yoy_end=100*(s.bev.iloc[-1]/s.bev.iloc[-13]-1),
             phev_yoy_end=100*(s.phev.iloc[-1]/s.phev.iloc[-13]-1))
    m['annual'] = {
        date: dict(change=s.loc[date, 'yoy_delta'], pct=s.loc[date, 'yoy'])
        for date in ['2020-12-31', '2022-12-31', str(s.index[-1].date())]}
    minimum = s.delta.idxmin()
    m['minimum_month'] = dict(date=str(minimum.date()), delta=s.loc[minimum, 'delta'],
                              mom=s.loc[minimum, 'mom'], value=s.loc[minimum, 'ev_total'],
                              previous=s.ev_total.shift().loc[minimum])
    before = s.index[s.index.get_loc(minimum)-1]
    m['before_minimum'] = dict(date=str(before.date()), delta=s.loc[before, 'delta'], mom=s.loc[before, 'mom'])
    m['top_months'] = [dict(date=str(date.date()), delta=s.loc[date, 'delta'], mom=s.loc[date, 'mom'])
                       for date in s.delta.nlargest(3).index]
    m['source_sha256'] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    (output_dir/'report_metrics.json').write_text(
        json.dumps(clean_json(m), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    # Bản Word sẽ từ chối dùng hình/CSV khác với bộ kết quả đã tạo cùng lần chạy.
    files = [p for p in output_dir.iterdir()
             if p.name not in {'artifact_manifest.json', 'report_claims.csv'}
             and p.suffix in {'.csv', '.json', '.png'}]
    manifest = dict(source_sha256=m['source_sha256'],
                    analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    versions={name: importlib.metadata.version(name)
                              for name in ['numpy', 'pandas', 'matplotlib', 'statsmodels']},
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
    (output_dir/'artifact_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="Đường dẫn CSV History đã tiền xử lý.")
    parser.add_argument("--output", type=Path, default=root / "analysis", help="Thư mục kết quả.")
    args = parser.parse_args()
    input_path = args.input
    if input_path is None:
        candidates = [root / "history.csv", root / "EV_Population_History_Washington_Clean.csv"]
        input_path = next((p for p in candidates if p.is_file()), None)
        if input_path is None:
            parser.error("Không thấy CSV bên cạnh code. Giải nén cả gói ZIP hoặc dùng --input đường_dẫn_CSV.")
    if not input_path.is_file():
        parser.error(f"Không tìm thấy file: {input_path}")
    try:
        series = load_series(input_path)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if input_path.resolve().parent == output_dir and input_path.name in {
        "ch4_2_series.csv", "notable_months.csv", "table4_1_growth.csv",
        "table4_2_variability.csv", "acf_values.csv", "component_growth.csv",
        "annual_growth.csv", "seasonal_growth_matrix.csv"}:
        parser.error("Thư mục đầu ra sẽ ghi đè CSV đầu vào. Hãy chọn --output khác.")
    summary = summarize(series)
    export_tables(series, summary, output_dir)
    make_figures(series, output_dir)
    export_report_metrics(series, summary, input_path, output_dir)
    print(f"Hoàn tất: {len(series)} tháng; 5 hình; kết quả tại {output_dir}")


if __name__ == "__main__":
    main()
