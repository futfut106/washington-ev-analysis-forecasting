"""
Script: src/generate_docx_tv2.py
Tác giả: Nguyễn Nữ Trà Giang (TV2) - Nhóm 8 - IS403.R14
Mô tả: Xuất báo cáo Word Mục 5.2 chuẩn học thuật, đồng bộ định dạng báo cáo Nhóm 8,
      sử dụng Subscript/Superscript thật của Word, không lỗi hiển thị ký hiệu.
"""

from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls


def create_report():
    doc = docx.Document()

    # Thiết lập trang A4 chuẩn theo báo cáo Nhóm 8
    for section in doc.sections:
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.top_margin = Inches(0.80)
        section.bottom_margin = Inches(0.80)
        section.left_margin = Inches(0.90)
        section.right_margin = Inches(0.90)

    FONT_NAME = "Times New Roman"
    BODY_SIZE = Pt(12)
    TABLE_SIZE = Pt(10.5)
    CAPTION_SIZE = Pt(11)
    COLOR_BLACK = RGBColor(0x00, 0x00, 0x00)
    COLOR_PRIMARY = RGBColor(0x1F, 0x4E, 0x79)

    def format_paragraph(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=6, line_spacing=1.3):
        p.alignment = align
        pf = p.paragraph_format
        pf.space_before = Pt(before)
        pf.space_after = Pt(after)
        pf.line_spacing = line_spacing
        return p

    def add_run_styled(p, text, size=BODY_SIZE, bold=False, italic=False, sub=False, sup=False, color=COLOR_BLACK):
        r = p.add_run(text)
        r.font.name = FONT_NAME
        r.font.size = size
        r.bold = bold
        r.italic = italic
        r.font.subscript = sub
        r.font.superscript = sup
        r.font.color.rgb = color
        return r

    def add_rich_p(segments, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=6, line_spacing=1.3, size=BODY_SIZE):
        """
        segments: list of tuples hoặc str:
            - str: văn bản thường
            - (text, style_str): style_str chứa 'b' (bold), 'i' (italic), 'sub' (subscript), 'sup' (superscript)
        """
        p = doc.add_paragraph()
        format_paragraph(p, align=align, before=before, after=after, line_spacing=1.3)
        for seg in segments:
            if isinstance(seg, str):
                add_run_styled(p, seg, size=size)
            else:
                txt, st = seg[0], seg[1]
                add_run_styled(
                    p,
                    txt,
                    size=size,
                    bold=("b" in st),
                    italic=("i" in st),
                    sub=("sub" in st),
                    sup=("sup" in st),
                )
        return p

    def add_heading(text, level=2):
        p = doc.add_paragraph()
        if level == 1:
            format_paragraph(p, align=WD_ALIGN_PARAGRAPH.LEFT, before=6, after=6, line_spacing=1.3)
            add_run_styled(p, text, size=Pt(13.5), bold=True, color=COLOR_BLACK)
        else:
            format_paragraph(p, align=WD_ALIGN_PARAGRAPH.LEFT, before=8, after=4, line_spacing=1.3)
            add_run_styled(p, text, size=Pt(12.5), bold=True, color=COLOR_BLACK)
        return p

    def add_formula_line(segments):
        p = doc.add_paragraph()
        format_paragraph(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=4, after=6, line_spacing=1.2)
        for seg in segments:
            if isinstance(seg, str):
                add_run_styled(p, seg, size=BODY_SIZE)
            else:
                txt, st = seg[0], seg[1]
                add_run_styled(
                    p,
                    txt,
                    size=BODY_SIZE,
                    bold=("b" in st),
                    italic=("i" in st),
                    sub=("sub" in st),
                    sup=("sup" in st),
                )
        return p

    def style_table(table, col_widths):
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        tblPr = table._tbl.tblPr
        borders = parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'<w:top w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
            f'<w:left w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
            f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
            f'<w:right w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
            f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
            f'<w:insideV w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
            f'</w:tblBorders>'
        )
        tblPr.append(borders)

        for r_idx, row in enumerate(table.rows):
            for c_idx, cell in enumerate(row.cells):
                cell.width = col_widths[c_idx]
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                tcPr = cell._tc.get_or_add_tcPr()
                tcMar = parse_xml(
                    f'<w:tcMar {nsdecls("w")}>'
                    f'<w:top w:w="60" w:type="dxa"/>'
                    f'<w:bottom w:w="60" w:type="dxa"/>'
                    f'<w:left w:w="90" w:type="dxa"/>'
                    f'<w:right w:w="90" w:type="dxa"/>'
                    f'</w:tcMar>'
                )
                tcPr.append(tcMar)
                if r_idx == 0:
                    shd = parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="D9E2F3"/>')
                    tcPr.append(shd)

    def fill_cell(cell, segments, align=WD_ALIGN_PARAGRAPH.LEFT, is_header=False):
        p = cell.paragraphs[0]
        p.text = ""
        format_paragraph(p, align=align, before=2, after=2, line_spacing=1.15)
        for seg in segments:
            if isinstance(seg, str):
                add_run_styled(p, seg, size=TABLE_SIZE, bold=is_header)
            else:
                txt, st = seg[0], seg[1]
                add_run_styled(
                    p,
                    txt,
                    size=TABLE_SIZE,
                    bold=(is_header or ("b" in st)),
                    italic=("i" in st),
                    sub=("sub" in st),
                    sup=("sup" in st),
                )

    # =========================================================================
    # TIÊU ĐỀ MỤC 5.2
    # =========================================================================
    add_heading("5.2. Đặc trưng và thiết kế thực nghiệm", level=1)

    add_rich_p([
        "Mục này thiết lập biểu diễn biến mục tiêu, hệ thống đặc trưng trễ và khung kiểm định chéo theo thời gian (time-series backtesting) dùng chung cho cả bốn mô hình học máy ở Chương 5 gồm Ridge (mốc tham chiếu tuyến tính), Random Forest, Support Vector Regression (SVR) và XGBoost. Toàn bộ thiết kế giữ nguyên chuỗi lịch sử toàn bang ",
        ("ch4_series.csv", "i"),
        " được tổng hợp từ bộ dữ liệu [H] [5] gồm 83 tháng (01/2017–11/2023), với 71 tháng đầu (01/2017–11/2022) làm tập huấn luyện, gốc dự báo cố định tại tháng 11/2022 và chân trời dự báo 12 tháng (12/2022–11/2023) như Chương 4."
    ])

    # =========================================================================
    # 5.2.1
    # =========================================================================
    add_heading("5.2.1. Biểu diễn chuỗi tăng quy mô và biến mục tiêu tăng trưởng log", level=2)

    add_rich_p([
        "Trong giai đoạn huấn luyện (01/2017–11/2022), tổng số xe điện đang đăng ký (",
        ("EV Total", "i"),
        ", ký hiệu ",
        ("y", "i"), ("t", "i,sub"),
        ") tăng liên tục từ 21.795 xe lên 112.948 xe và tiếp tục tăng lên 159.478 xe vào tháng 11/2023. Nếu huấn luyện trực tiếp các mô hình cây hồi quy (Random Forest, XGBoost) trên mức quy mô ",
        ("y", "i"), ("t", "i,sub"),
        ", dự báo của mô hình chỉ là trung bình có trọng số của các quan sát nằm trong các nút lá nên bị chặn trên bởi giá trị lớn nhất từng xuất hiện trong tập huấn luyện (",
        ("y", "i"), ("max", "sub"), ("train", "sup"),
        " = 112.948 xe) [1], [2]. Mặt khác, kết quả kiểm định ADF ở Chương 4 (chọn ",
        ("d", "i"), " = 2 cho ARIMA) là điều kiện dừng tuyến tính của họ mô hình tự hồi quy thống kê, không phải điều kiện bắt buộc để áp đặt sai phân bậc hai một cách máy móc cho mọi thuật toán học máy phi tuyến."
    ])

    add_rich_p([
        "Để xử lý xu hướng tăng quy mô mà vẫn giữ khả năng ngoại suy qua 12 bước, nghiên cứu đề xuất cho cả bốn mô hình học trên tốc độ tăng trưởng log tháng ",
        ("r", "i"), ("t", "i,sub"),
        " đã chuẩn hóa theo từng fold [1], [3], [4]:"
    ])

    add_formula_line([
        ("r", "i"), ("t", "i,sub"),
        " = log(", ("y", "i"), ("t", "i,sub"), ") − log(", ("y", "i"), ("t−1", "i,sub"), ") = log(",
        ("y", "i"), ("t", "i,sub"), " / ", ("y", "i"), ("t−1", "i,sub"), "),      ",
        ("z", "i"), ("t", "i,sub"),
        " = (", ("r", "i"), ("t", "i,sub"), " − ", ("μ", "i"), ("r,train", "sub"), ") / ",
        ("σ", "i"), ("r,train", "sub"),
        "    (5.1)"
    ])

    add_rich_p([
        "Trong đó ",
        ("μ", "i"), ("r,train", "sub"),
        " và ",
        ("σ", "i"), ("r,train", "sub"),
        " là trung bình và độ lệch chuẩn của nhãn ",
        ("r", "i"), ("t", "i,sub"),
        " được ước lượng riêng trên tập huấn luyện của từng fold. Trên 70 tháng có tăng trưởng của tập huấn luyện (02/2017–11/2022), ",
        ("r", "i"), ("t", "i,sub"),
        " dao động ổn định quanh trung bình 0,02350 (độ lệch chuẩn 0,00844; thấp nhất −0,00016 vào tháng 05/2020 và cao nhất 0,04519); trên 64 cặp học có giám sát (08/2017–11/2022), ",
        ("μ", "i"), ("r,train", "sub"), " = 0,02321 và ",
        ("σ", "i"), ("r,train", "sub"), " = 0,00874. Khi dự báo từ gốc ",
        ("T", "i"), " với chân trời ", ("h", "i"), " ∈ {1, …, 12}, mô hình sinh dự báo ",
        ("ẑ", "i"), ("T+h", "i,sub"),
        ", đổi ngược về tốc độ tăng trưởng ",
        ("r̂", "i"), ("T+h", "i,sub"), " = ", ("μ", "i"), ("r,train", "sub"), " + ", ("σ", "i"), ("r,train", "sub"), ("ẑ", "i"), ("T+h", "i,sub"),
        ", rồi tích lũy về quy mô xe theo công thức:"
    ])

    add_formula_line([
        ("ŷ", "i"), ("T+h", "i,sub"),
        " = ", ("y", "i"), ("T", "i,sub"), " × exp(",
        ("Σ", "b"), ("j=1..h", "sub"), " ", ("r̂", "i"), ("T+j", "i,sub"),
        ")    (5.2)"
    ])

    add_rich_p([
        "Cần nhấn mạnh ba nguyên tắc phương pháp luận trong biểu diễn này: (i) việc học ",
        ("r", "i"), ("t", "i,sub"),
        " rồi tích lũy lại ",
        ("y", "i"), ("t", "i,sub"),
        " là một giả thuyết thiết kế nhằm hỗ trợ các mô hình cây vượt trần tập huấn luyện, cần được kiểm tra thực nghiệm chứ không mặc định đảm bảo cải thiện so với SARIMA; (ii) biến đích khi chấm điểm và báo cáo cuối cùng vẫn là số lượng xe ",
        ("EV Total", "i"),
        "; và (iii) nhóm không mặc nhiên gọi phép biến đổi hàm mũ ở công thức (5.2) là kỳ vọng toán đã hiệu chỉnh sai lệch, đồng thời không kẹp cơ học các dự báo tăng trưởng âm thành 0 (thực tế dữ liệu lịch sử từng ghi nhận ",
        ("r", "i"), ("05/2020", "sub"), " = −0,00016)."
    ])

    # =========================================================================
    # 5.2.2
    # =========================================================================
    add_heading("5.2.2. Thiết kế năm đặc trưng quá khứ và kiểm soát rò rỉ dữ liệu", level=2)

    add_rich_p([
        "Do quy mô mẫu theo tháng nhỏ, nhóm giới hạn không gian đầu vào gồm đúng 5 đặc trưng quá khứ của chuỗi ",
        ("r", "i"), ("t", "i,sub"),
        " để tránh quá khớp (overfitting) [1], [3]: ba độ trễ trực tiếp (",
        ("r", "i"), ("t−1", "i,sub"), ", ",
        ("r", "i"), ("t−2", "i,sub"), ", ",
        ("r", "i"), ("t−3", "i,sub"),
        ") phản ánh động lượng ngắn hạn và hai trung bình trượt (",
        ("r̄", "i"), ("t−1:t−3", "i,sub"), ", ",
        ("r̄", "i"), ("t−1:t−6", "i,sub"),
        ") nắm bắt xu thế tăng trưởng trung bình quý và nửa năm. Để loại bỏ hoàn toàn hiện tượng rò rỉ thông tin hiện tại vào đặc trưng (look-ahead bias), toàn bộ phép tính trung bình trượt bắt buộc phải thực hiện dịch trễ một bước ",
        ("shift(1)", "b"),
        " trước khi lấy cửa sổ ",
        ("rolling(3)", "i"), " và ", ("rolling(6)", "i"),
        " [3], [4]. Nhờ đó, tại thời điểm ", ("t", "i"),
        ", vector đặc trưng ", ("X", "i"), ("t", "i,sub"),
        " chỉ phụ thuộc vào tập lịch sử {", ("r", "i"), ("t−1", "i,sub"), ", …, ", ("r", "i"), ("t−6", "i,sub"), "}."
    ])

    add_rich_p([
        "Vì đặc trưng trung bình trượt 6 tháng cần 6 giá trị ",
        ("r", "i"),
        " quá khứ (tương ứng 7 tháng mức xe ",
        ("y", "i"),
        " từ tháng 01/2017 đến tháng 07/2017), 71 tháng huấn luyện tạo ra đúng 64 cặp học có giám sát từ tháng 08/2017 đến tháng 11/2022. Nhóm không cắt bớt 7 tháng đầu khỏi ARIMA/SARIMA để ép số dòng bằng nhau: mọi mô hình đều được tiếp cận cùng lịch sử đến tháng 11/2022, nhưng số quan sát hữu dụng sau sai phân hoặc tạo độ trễ khác nhau và được công bố rõ trong bảng kết quả. Bảng 5.1 trình bày từ điển 5 đặc trưng cùng giá trị minh họa tại dòng học đầu tiên (tháng 08/2017)."
    ])

    # Bảng 5.1
    add_rich_p(
        [("Bảng 5.1. Từ điển năm đặc trưng quá khứ và giá trị minh họa tại dòng đầu tiên (08/2017)", "b")],
        align=WD_ALIGN_PARAGRAPH.CENTER, before=4, after=3
    )

    t1 = doc.add_table(rows=6, cols=6)
    w1 = [Inches(1.05), Inches(1.35), Inches(0.65), Inches(1.65), Inches(0.95), Inches(0.82)]
    style_table(t1, w1)

    fill_cell(t1.rows[0].cells[0], ["Tên cột mã"], align=WD_ALIGN_PARAGRAPH.CENTER, is_header=True)
    fill_cell(t1.rows[0].cells[1], ["Công thức toán"], align=WD_ALIGN_PARAGRAPH.CENTER, is_header=True)
    fill_cell(t1.rows[0].cells[2], ["Cửa sổ"], align=WD_ALIGN_PARAGRAPH.CENTER, is_header=True)
    fill_cell(t1.rows[0].cells[3], ["Ý nghĩa kinh tế – thống kê"], align=WD_ALIGN_PARAGRAPH.CENTER, is_header=True)
    fill_cell(t1.rows[0].cells[4], ["Chuẩn hóa X"], align=WD_ALIGN_PARAGRAPH.CENTER, is_header=True)
    fill_cell(t1.rows[0].cells[5], ["Giá trị 08/2017"], align=WD_ALIGN_PARAGRAPH.CENTER, is_header=True)

    rows_t1 = [
        (
            [("r_lag1", "b"), " (r_lag_1)"],
            [("r", "i"), ("t−1", "i,sub")],
            ["Trễ 1"],
            ["Tăng trưởng log tháng liền trước (07/2017)"],
            ["Ridge, SVR: Có; RF, XGB: Không"],
            ["0,022424"]
        ),
        (
            [("r_lag2", "b"), " (r_lag_2)"],
            [("r", "i"), ("t−2", "i,sub")],
            ["Trễ 2"],
            ["Tăng trưởng log cách 2 tháng (06/2017)"],
            ["Ridge, SVR: Có; RF, XGB: Không"],
            ["0,026505"]
        ),
        (
            [("r_lag3", "b"), " (r_lag_3)"],
            [("r", "i"), ("t−3", "i,sub")],
            ["Trễ 3"],
            ["Tăng trưởng log cách 3 tháng (05/2017)"],
            ["Ridge, SVR: Có; RF, XGB: Không"],
            ["0,025329"]
        ),
        (
            [("r_roll3", "b"), " (r_mean_3)"],
            ["(", ("r", "i"), ("t−1", "i,sub"), " + ", ("r", "i"), ("t−2", "i,sub"), " + ", ("r", "i"), ("t−3", "i,sub"), ") / 3"],
            ["3 tháng (shift 1)"],
            ["Trung bình trượt quý gần nhất (05–07/2017)"],
            ["Ridge, SVR: Có; RF, XGB: Không"],
            ["0,024753"]
        ),
        (
            [("r_roll6", "b"), " (r_mean_6)"],
            ["(", ("r", "i"), ("t−1", "i,sub"), " + … + ", ("r", "i"), ("t−6", "i,sub"), ") / 6"],
            ["6 tháng (shift 1)"],
            ["Trung bình trượt nửa năm trước (02–07/2017)"],
            ["Ridge, SVR: Có; RF, XGB: Không"],
            ["0,026623"]
        ),
    ]

    for idx, rdata in enumerate(rows_t1, start=1):
        fill_cell(t1.rows[idx].cells[0], rdata[0], align=WD_ALIGN_PARAGRAPH.LEFT)
        fill_cell(t1.rows[idx].cells[1], rdata[1], align=WD_ALIGN_PARAGRAPH.CENTER)
        fill_cell(t1.rows[idx].cells[2], rdata[2], align=WD_ALIGN_PARAGRAPH.CENTER)
        fill_cell(t1.rows[idx].cells[3], rdata[3], align=WD_ALIGN_PARAGRAPH.LEFT)
        fill_cell(t1.rows[idx].cells[4], rdata[4], align=WD_ALIGN_PARAGRAPH.CENTER)
        fill_cell(t1.rows[idx].cells[5], rdata[5], align=WD_ALIGN_PARAGRAPH.RIGHT)

    add_rich_p(
        [("Nguồn: tính từ [5], trích dòng đầu tiên của tệp kiểm tra supervised_train.csv.", "i")],
        align=WD_ALIGN_PARAGRAPH.CENTER, before=2, after=6, size=CAPTION_SIZE
    )

    add_rich_p([
        ("Ví dụ kiểm tra một dòng (tháng 08/2017): ", "b"),
        "Từ 7 tháng lịch sử đầu tiên của chuỗi mức xe ",
        ("y", "i"), ("01/2017..07/2017", "sub"),
        " = {21.795; 22.470; 23.062; 23.740; 24.349; 25.003; 25.570}, ta thu được 6 giá trị tăng trưởng log quá khứ ",
        ("r", "i"), ("02/2017..07/2017", "sub"),
        " = {0,030501; 0,026005; 0,028975; 0,025329; 0,026505; 0,022424}. Tại tháng ",
        ("t", "i"), " = 08/2017 (", ("y", "i"), ("t", "i,sub"), " = 26.070 xe), nhãn thực tế là ",
        ("r", "i"), ("08/2017", "sub"), " = log(26.070 / 25.570) = 0,019365 (tương ứng ",
        ("z", "i"), ("08/2017", "sub"), " = −0,4401 khi chuẩn hóa trên 64 dòng train), trong khi 5 đặc trưng đầu vào tại Bảng 5.1 chỉ được tính từ các tháng 02/2017–07/2017, hoàn toàn không chứa thông tin của tháng 08/2017."
    ])

    # =========================================================================
    # 5.2.3
    # =========================================================================
    add_heading("5.2.3. Thiết kế kiểm định chéo theo thời gian (Validation Folds) bên trong tập huấn luyện", level=2)

    add_rich_p([
        "Để lựa chọn siêu tham số hợp lệ cho chuỗi thời gian có dự báo đệ quy, nhóm không dùng K-fold ngẫu nhiên và không dùng ",
        ("GridSearchCV", "i"),
        " một bước làm phép đánh giá thay thế [4], [7]. Lý do là ",
        ("GridSearchCV", "i"),
        " tiêu chuẩn chỉ đo sai số dự báo 1 bước (", ("h", "i"),
        " = 1) khi mọi độ trễ đều lấy từ giá trị thật, không phản ánh hiện tượng sai số bước trước đi vào đặc trưng bước sau trong chân trời 12 tháng. Thay vào đó, nhóm thiết lập 3 fold mở rộng (expanding window) nằm trọn trong 71 tháng huấn luyện (Bảng 5.2)."
    ])

    # Bảng 5.2
    add_rich_p(
        [("Bảng 5.2. Thiết kế ba fold kiểm định chéo theo thời gian bên trong tập huấn luyện", "b")],
        align=WD_ALIGN_PARAGRAPH.CENTER, before=4, after=3
    )

    t2 = doc.add_table(rows=5, cols=6)
    w2 = [Inches(0.75), Inches(1.30), Inches(1.45), Inches(1.15), Inches(0.90), Inches(0.92)]
    style_table(t2, w2)

    fill_cell(t2.rows[0].cells[0], ["Cửa sổ"], align=WD_ALIGN_PARAGRAPH.CENTER, is_header=True)
    fill_cell(t2.rows[0].cells[1], ["Train đến (Origin)"], align=WD_ALIGN_PARAGRAPH.CENTER, is_header=True)
    fill_cell(t2.rows[0].cells[2], ["Kỳ dự báo 12 tháng"], align=WD_ALIGN_PARAGRAPH.CENTER, is_header=True)
    fill_cell(t2.rows[0].cells[3], ["Tháng train / Cặp học"], align=WD_ALIGN_PARAGRAPH.CENTER, is_header=True)
    fill_cell(t2.rows[0].cells[4], ["Gốc ", ("y", "i"), ("T", "i,sub"), " (xe)"], align=WD_ALIGN_PARAGRAPH.CENTER, is_header=True)
    fill_cell(t2.rows[0].cells[5], [("μ", "i"), ("r", "sub"), " / ", ("σ", "i"), ("r", "sub"), " (train)"], align=WD_ALIGN_PARAGRAPH.CENTER, is_header=True)

    rows_t2 = [
        (["Fold 1"], ["11/2020 (2020-11-30)"], ["12/2020–11/2021 (Val)"], ["47 / 40"], ["63.519"], ["0,02275 / 0,01027"]),
        (["Fold 2"], ["05/2021 (2021-05-31)"], ["06/2021–05/2022 (Val)"], ["53 / 46"], ["72.726"], ["0,02272 / 0,00984"]),
        (["Fold 3"], ["11/2021 (2021-11-30)"], ["12/2021–11/2022 (Val)"], ["59 / 52"], ["84.440"], ["0,02297 / 0,00938"]),
        ([("Full Train", "b")], [("11/2022 (2022-11-30)", "b")], [("12/2022–11/2023 (Test)", "b")], [("71 / 64", "b")], [("112.948", "b")], [("0,02321 / 0,00874", "b")]),
    ]

    for idx, rdata in enumerate(rows_t2, start=1):
        fill_cell(t2.rows[idx].cells[0], rdata[0], align=WD_ALIGN_PARAGRAPH.CENTER)
        fill_cell(t2.rows[idx].cells[1], rdata[1], align=WD_ALIGN_PARAGRAPH.CENTER)
        fill_cell(t2.rows[idx].cells[2], rdata[2], align=WD_ALIGN_PARAGRAPH.CENTER)
        fill_cell(t2.rows[idx].cells[3], rdata[3], align=WD_ALIGN_PARAGRAPH.CENTER)
        fill_cell(t2.rows[idx].cells[4], rdata[4], align=WD_ALIGN_PARAGRAPH.RIGHT)
        fill_cell(t2.rows[idx].cells[5], rdata[5], align=WD_ALIGN_PARAGRAPH.CENTER)

    add_rich_p(
        [("Nguồn: tính từ [5], xuất trong tệp folds.csv và protocol.json.", "i")],
        align=WD_ALIGN_PARAGRAPH.CENTER, before=2, after=6, size=CAPTION_SIZE
    )

    add_rich_p([
        "Mỗi cấu hình ứng viên phải chạy đủ 3 fold và dự báo đệ quy đủ 12 bước ở từng fold. Tiêu chí chọn cấu hình là trung bình RMSE qua 3 fold tính trên thang số lượng xe (báo cáo kèm MAE, sMAPE và Bias). Nếu có cấu hình đơn giản hơn đạt RMSE nằm trong phạm vi 1% so với mức tốt nhất, nhóm chọn cấu hình đơn giản hơn theo quy tắc ghi trước. Do các cửa sổ validation dịch chuyển cách nhau 6 tháng nên có phần chồng lấn (06/2021–11/2021 và 12/2021–05/2022), kết quả 3 fold không được coi là các mẫu độc lập để kiểm định ý nghĩa thống kê. Sau khi khóa cấu hình từ validation, mô hình được fit lại một lần trên 71 tháng train (64 cặp học) để đánh giá trên 12 tháng cuối (12/2022–11/2023). Vì giai đoạn này đã được quan sát ở Chương 4, nhóm gọi đây là cửa sổ so sánh ngoài mẫu cố định (fixed comparison window), không tuyên bố là tập kiểm tra mù hoàn toàn và tuyệt đối không chỉnh sửa đặc trưng hay siêu tham số sau khi xem điểm test."
    ])

    # =========================================================================
    # 5.2.4
    # =========================================================================
    add_heading("5.2.4. Cơ chế dự báo đệ quy 12 bước, giao diện dùng chung và kiểm thử không rò rỉ", level=2)

    add_rich_p([
        "Trong tệp ", ("common/features.py", "i"), " và ", ("common/backtest.py", "i"),
        ", giao diện huấn luyện và dự báo được chuẩn hóa thống nhất cho cả bốn mô hình thông qua hai hàm chính: ",
        ("fit(train_history, config)", "b"), " → ", ("artifact", "i"),
        " và ",
        ("forecast(artifact, train_history, horizon=12)", "b"),
        ". Tệp ", ("supervised_train.csv", "i"),
        " (64 dòng) chỉ là bản xuất để đối chiếu; khi chạy backtest, từng fold tự cắt chuỗi gốc tại mốc ",
        ("origin", "i"),
        ", tự tạo lại bảng học (40/46/52 dòng) và tự fit lại scaler của ",
        ("X", "i"), " (với Ridge, SVR) cùng scaler của nhãn ", ("z", "i"),
        " bên trong fold đó [4], [6]. Khi dự báo đệ quy 12 bước, bộ đệm (buffer) chỉ khởi tạo bằng 6 giá trị ",
        ("r", "i"), " lịch sử cuối cùng của tập train; từ bước ", ("h", "i"), " = 2 trở đi, các độ trễ và trung bình trượt chỉ được cập nhật bằng chính dự báo ",
        ("r̂", "i"), ("T+h−1", "i,sub"),
        " do mô hình tự sinh (Hình 5.1)."
    ])

    # Chèn Hình 5.1
    fig_img_path = Path("D:/Giang/IS403/IS403_repo/washington-ev-analysis-forecasting/figures/ch5_tv2_workflow.png")
    if fig_img_path.exists():
        p_fig = doc.add_paragraph()
        format_paragraph(p_fig, align=WD_ALIGN_PARAGRAPH.CENTER, before=4, after=2, line_spacing=1.0)
        p_fig.add_run().add_picture(str(fig_img_path), width=Inches(6.45))

        add_rich_p(
            [("Hình 5.1. Sơ đồ quy trình từ chuỗi lịch sử đến đặc trưng, huấn luyện, dự báo đệ quy 12 bước và quy đổi về số xe", "b")],
            align=WD_ALIGN_PARAGRAPH.CENTER, before=2, after=1, size=CAPTION_SIZE
        )
        add_rich_p(
            [("Nguồn: Nhóm xây dựng.", "i")],
            align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=6, size=CAPTION_SIZE
        )

    add_rich_p([
        "Để xác minh ranh giới thời gian bằng kiểm thử tự động, tệp ",
        ("test_leakage.py", "i"),
        " triển khai 5 bài kiểm thử đơn vị (Bảng 5.3). Toàn bộ 5 bài kiểm thử đều vượt qua (PASS 100%), chứng minh dự báo của mô hình đã huấn luyện hoàn toàn bất biến khi thay mọi giá trị validation/test bằng số giả, đồng thời đảm bảo các thành viên mô hình (TV1, TV3, TV4, TV5) dùng chung một giao diện mà không phải sao chép lại logic tạo đặc trưng."
    ])

    # Bảng 5.3
    add_rich_p(
        [("Bảng 5.3. Tóm tắt bộ kiểm thử đơn vị về ranh giới thời gian và chống rò rỉ dữ liệu (test_leakage.py)", "b")],
        align=WD_ALIGN_PARAGRAPH.CENTER, before=4, after=3
    )

    t3 = doc.add_table(rows=6, cols=4)
    w3 = [Inches(1.55), Inches(2.25), Inches(1.85), Inches(0.82)]
    style_table(t3, w3)

    fill_cell(t3.rows[0].cells[0], ["Hàm kiểm thử"], align=WD_ALIGN_PARAGRAPH.CENTER, is_header=True)
    fill_cell(t3.rows[0].cells[1], ["Điều kiện kiểm tra"], align=WD_ALIGN_PARAGRAPH.CENTER, is_header=True)
    fill_cell(t3.rows[0].cells[2], ["Tiêu chuẩn nghiệm thu"], align=WD_ALIGN_PARAGRAPH.CENTER, is_header=True)
    fill_cell(t3.rows[0].cells[3], ["Kết quả"], align=WD_ALIGN_PARAGRAPH.CENTER, is_header=True)

    rows_t3 = [
        (["test_01_sample_sizes"], ["Kích thước tập học của 3 fold và full train"], ["Đúng 40 / 46 / 52 và 64 dòng (từ 08/2017)"], [("Đạt (Pass)", "b")]),
        (["test_02_shift1_enforcement"], ["Thay đổi ", ("r", "i"), ("t", "i,sub"), " = 999,99 tại thời điểm hiện tại ", ("t", "i")], ["Vector đặc trưng ", ("X", "i"), ("t", "i,sub"), " không thay đổi"], [("Đạt (Pass)", "b")]),
        (["test_03_zero_future_leakage"], ["Thay toàn bộ validation/test bằng số giả ngẫu nhiên"], ["12 dự báo ", ("ŷ", "i"), ("T+h", "i,sub"), " trùng khớp tuyệt đối; điểm số thay đổi"], [("Đạt (Pass)", "b")]),
        (["test_04_mathematical_inversion"], ["Kiểm tra tích lũy ", ("ẑ", "i"), " → ", ("r̂", "i"), " → ", ("ŷ", "i"), " cả khi ", ("r̂", "i"), " < 0"], ["Khớp công thức (5.2), không kẹp âm về 0"], [("Đạt (Pass)", "b")]),
        (["test_05_unified_api_models"], ["Chạy cùng API cho Ridge, RF, SVR và XGBoost"], ["Sinh đủ 12 dự báo hữu hạn và bảng trace 12 bước"], [("Đạt (Pass)", "b")]),
    ]

    for idx, rdata in enumerate(rows_t3, start=1):
        fill_cell(t3.rows[idx].cells[0], rdata[0], align=WD_ALIGN_PARAGRAPH.LEFT)
        fill_cell(t3.rows[idx].cells[1], rdata[1], align=WD_ALIGN_PARAGRAPH.LEFT)
        fill_cell(t3.rows[idx].cells[2], rdata[2], align=WD_ALIGN_PARAGRAPH.LEFT)
        fill_cell(t3.rows[idx].cells[3], rdata[3], align=WD_ALIGN_PARAGRAPH.CENTER)

    add_rich_p(
        [("Nguồn: kết quả thực thi trực tiếp bộ kiểm thử test_leakage.py.", "i")],
        align=WD_ALIGN_PARAGRAPH.CENTER, before=2, after=4, size=CAPTION_SIZE
    )

    # Save docx
    out_report = Path("D:/Giang/IS403/IS403_repo/washington-ev-analysis-forecasting/reports/Muc_5.2_Dac_trung_va_Thiet_ke_thuc_nghiem.docx")
    doc.save(out_report)
    print("Document successfully saved to:", out_report)

    out_parent = Path("D:/Giang/IS403/Muc_5.2_TV2_Dac_trung_va_khung_thuc_nghiem.docx")
    doc.save(out_parent)
    print("Copy successfully saved to:", out_parent)


if __name__ == "__main__":
    create_report()
