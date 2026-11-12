"""
Script: src/generate_flowchart_tv2.py
Mô tả: Tạo sơ đồ quy trình thực nghiệm Mục 5.2 (Hình 5.1) rõ nét, không tràn chữ.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_workflow_figure():
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(13.8, 6.0), dpi=300)
    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    c_box = "#F7F9FC"
    c_header = "#D9E2F3"
    c_border = "#1F4E79"
    c_accent = "#2E75B6"
    c_text = "#1A1A1A"

    def add_step_box(ax, x, y, w, h, title, lines):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.10",
            facecolor=c_box, edgecolor=c_border, linewidth=1.5, zorder=2
        )
        ax.add_patch(rect)
        hdr_h = 0.44
        hdr = patches.FancyBboxPatch(
            (x, y + h - hdr_h), w, hdr_h,
            boxstyle="round,pad=0.05,rounding_size=0.10",
            facecolor=c_header, edgecolor=c_border, linewidth=1.2, zorder=3
        )
        ax.add_patch(hdr)
        ax.text(
            x + w / 2, y + h - hdr_h / 2, title,
            ha="center", va="center", fontsize=8.8, fontweight="bold", color=c_border, zorder=4
        )
        body_text = "\n".join(lines)
        ax.text(
            x + w / 2, y + (h - hdr_h) / 2 - 0.01, body_text,
            ha="center", va="center", fontsize=7.9, color=c_text,
            linespacing=1.52, multialignment="center", zorder=4
        )

    def add_h_arrow(ax, x1, x2, y, label_top="", label_bot=""):
        ax.annotate(
            "", xy=(x2, y), xytext=(x1, y),
            arrowprops=dict(arrowstyle="->,head_width=0.32,head_length=0.42", color=c_accent, lw=2.0),
            zorder=5
        )
        mx = (x1 + x2) / 2
        if label_top:
            ax.text(mx, y + 0.15, label_top, ha="center", va="bottom", fontsize=7.8, color="#1F4E79", fontweight="bold")
        if label_bot:
            ax.text(mx, y - 0.15, label_bot, ha="center", va="top", fontsize=7.4, color="#595959", style="italic")

    wb, hb = 3.40, 1.95
    y_top = 3.55
    y_bot = 0.45
    x1, x2, x3 = 0.25, 5.20, 10.15

    # Row 1: Left -> Right
    add_step_box(ax, x1, y_top, wb, hb, "1. Chuỗi lịch sử EV Total ($y_t$)", [
        "• Nguồn [H]: 83 tháng (01/2017–11/2023)",
        "• Train: 71 tháng đầu (01/2017–11/2022)",
        "• Test: 12 tháng cuối (12/2022–11/2023)",
        "• Gốc dự báo cố định: $T = 11/2022$"
    ])

    add_step_box(ax, x2, y_top, wb, hb, "2. Biểu diễn tăng trưởng ($r_t, z_t$)", [
        r"• Tăng trưởng log: $r_t = \log(y_t) - \log(y_{t-1})$",
        r"• Chuẩn hóa nhãn: $z_t = (r_t - \mu_r) / \sigma_r$",
        r"• $\mu_r, \sigma_r$ chỉ fit trên tập train từng fold",
        "• Ngăn chặn rò rỉ tham số tương lai"
    ])

    add_step_box(ax, x3, y_top, wb, hb, "3. Tạo 5 đặc trưng quá khứ ($X_t$)", [
        r"• 3 độ trễ: $r_{t-1},\; r_{t-2},\; r_{t-3}$",
        r"• 2 TB trượt: $\bar{r}_{t-1:t-3},\; \bar{r}_{t-1:t-6}$",
        "• Bắt buộc shift(1) trước khi rolling",
        "• Cần 7 tháng đầu → 64 cặp học (08/2017+)"
    ])

    # Row 2: Right -> Left
    add_step_box(ax, x3, y_bot, wb, hb, "4. Huấn luyện mô hình 1 bước", [
        r"• Học ánh xạ: $X_t \rightarrow z_t$ trên từng fold",
        "• 3 fold CV: 40 / 46 / 52 cặp học",
        "• Ridge, SVR: chuẩn hóa $X$ trong pipeline",
        "• RF, XGBoost: giữ nguyên $X$; hàm fit()"
    ])

    add_step_box(ax, x2, y_bot, wb, hb, "5. Dự báo đệ quy 12 bước ($h = 1..12$)", [
        r"• Dự báo $\hat{z}_{T+h} \rightarrow \hat{r}_{T+h} = \mu_r + \sigma_r \hat{z}_{T+h}$",
        r"• Đẩy $\hat{r}_{T+h}$ vào buffer để tạo $X_{T+h+1}$",
        "• Không nhận giá trị thực tế tương lai",
        "• Giao diện dùng chung: hàm forecast()"
    ])

    add_step_box(ax, x1, y_bot, wb, hb, "6. Đổi về số xe & Đánh giá", [
        r"• Tích lũy: $\hat{y}_{T+h} = y_T \cdot \exp(\Sigma_{j=1}^h \hat{r}_{T+j})$",
        "• Không kẹp tăng trưởng âm thành 0",
        "• Đánh giá trên thang số lượng xe:",
        "  RMSE (chính), MAE, sMAPE, Bias"
    ])

    # Horizontal arrows top
    ym_top = y_top + hb / 2
    add_h_arrow(ax, x1 + wb + 0.08, x2 - 0.08, ym_top, "Sai phân log", "70 giá trị $r_t$")
    add_h_arrow(ax, x2 + wb + 0.08, x3 - 0.08, ym_top, "shift(1) & rolling", "Không nhìn $r_t$")

    # Vertical arrow right (Box 3 -> Box 4)
    vx = x3 + wb / 2
    ax.annotate(
        "", xy=(vx, y_bot + hb + 0.08), xytext=(vx, y_top - 0.08),
        arrowprops=dict(arrowstyle="->,head_width=0.32,head_length=0.42", color=c_accent, lw=2.0),
        zorder=5
    )
    ax.text(vx - 0.16, (y_top + y_bot + hb) / 2, "Tập học có giám sát", ha="right", va="center", fontsize=7.8, color="#1F4E79", fontweight="bold")
    ax.text(vx + 0.16, (y_top + y_bot + hb) / 2, "40 / 46 / 52 / 64 dòng", ha="left", va="center", fontsize=7.4, color="#595959", style="italic")

    # Horizontal arrows bottom (Right -> Left)
    ym_bot = y_bot + hb / 2
    add_h_arrow(ax, x3 - 0.08, x2 + wb + 0.08, ym_bot, "Artifact đã khóa", "model + scaler $X, z$")
    add_h_arrow(ax, x2 - 0.08, x1 + wb + 0.08, ym_bot, "Tích lũy hàm mũ", r"12 dự báo $\hat{y}_{T+h}$")

    fig_path = Path("D:/Giang/IS403/IS403_repo/washington-ev-analysis-forecasting/figures/ch5_tv2_workflow.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    print("Saved clean workflow diagram to:", fig_path)

if __name__ == "__main__":
    generate_workflow_figure()
