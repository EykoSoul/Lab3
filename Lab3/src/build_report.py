from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "output" / "pdf" / "report_decision_tree.pdf"

pdfmetrics.registerFont(TTFont("Arial", "/System/Library/Fonts/Supplemental/Arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(2 * cm, 1.45 * cm, 19 * cm, 1.45 * cm)
    canvas.setFont("Arial", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(2 * cm, 1.0 * cm, "Классификация с помощью дерева решений")
    canvas.drawRightString(19 * cm, 1.0 * cm, str(doc.page))
    canvas.restoreState()


def scaled_image(path: Path, max_width: float = 17 * cm, max_height: float = 9 * cm) -> Image:
    with PILImage.open(path) as image:
        width, height = image.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def fmt(value: float) -> str:
    return f"{value:.3f}".replace(".", ",")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    styles = getSampleStyleSheet()
    normal = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Arial", fontSize=10.5,
                            leading=15, alignment=TA_JUSTIFY, spaceAfter=7)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Arial-Bold", fontSize=19,
                        leading=23, textColor=colors.HexColor("#17324D"), spaceBefore=8, spaceAfter=12)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Arial-Bold", fontSize=14,
                        leading=18, textColor=colors.HexColor("#246B78"), spaceBefore=10, spaceAfter=8)
    caption = ParagraphStyle("Caption", parent=normal, fontSize=9, leading=12, alignment=TA_CENTER,
                             textColor=colors.HexColor("#475569"), spaceBefore=5, spaceAfter=10)
    small = ParagraphStyle("Small", parent=normal, fontSize=8.5, leading=11)
    code = ParagraphStyle("Code", parent=normal, fontName="Courier", fontSize=8.5, leading=11,
                          leftIndent=8, backColor=colors.HexColor("#F1F5F9"), borderPadding=6)
    bullet = ParagraphStyle("Bullet", parent=normal, leftIndent=15, firstLineIndent=-8, bulletIndent=4)

    doc = BaseDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                          topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                          title="Классификация с помощью дерева решений")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates(PageTemplate(id="default", frames=frame, onPage=footer))
    story = []

    story.extend([
        Spacer(1, 2.2 * cm),
        Paragraph("ОТЧЁТ ПО ЛАБОРАТОРНОЙ РАБОТЕ", ParagraphStyle("Kicker", parent=normal, alignment=TA_CENTER,
                  fontName="Arial-Bold", fontSize=12, textColor=colors.HexColor("#246B78"))),
        Spacer(1, 0.7 * cm),
        Paragraph("Классификация с помощью<br/>дерева решений", ParagraphStyle("Title", parent=h1,
                  alignment=TA_CENTER, fontSize=25, leading=31, spaceAfter=16)),
        Spacer(1, 0.6 * cm),
        Table([["Набор данных", "UCI Adult (Census Income)"],
               ["Целевая переменная", "Годовой доход: ≤ 50 000 или > 50 000 долларов"],
               ["Критерии", "Information Gain, Gain Ratio, Gini Index"],
               ["Дата формирования", "25 сентября 2026 г."]],
              colWidths=[5.0 * cm, 10.8 * cm], style=TableStyle([
                  ("FONT", (0, 0), (-1, -1), "Arial", 10),
                  ("FONT", (0, 0), (0, -1), "Arial-Bold", 10),
                  ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E7F1F4")),
                  ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B7CBD2")),
                  ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                  ("LEFTPADDING", (0, 0), (-1, -1), 8),
                  ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                  ("TOPPADDING", (0, 0), (-1, -1), 8),
                  ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
              ])),
        Spacer(1, 4.2 * cm),
        Paragraph("Исходный код, данные и результаты находятся в каталоге проекта lab3.",
                  ParagraphStyle("Center", parent=normal, alignment=TA_CENTER)),
        PageBreak(),
    ])

    story += [Paragraph("1. Формулировка задания", h1)]
    tasks = [
        "Разработать программу классификации набора данных деревом решений с критериями Information Gain, Gain Ratio и Gini Index.",
        "Провести эксперименты на Census Income, используя 100% исходной обучающей выборки для построения дерева.",
        "Визуализировать построенные деревья решений.",
        "Добавить параметр доли обучающей выборки и вычисление accuracy, precision, recall и F-меры.",
        "При фиксированном критерии провести эксперименты для соотношений 60:40, 70:30, 80:20 и 90:10.",
        "Построить диаграмму зависимости показателей качества от соотношения выборок и объяснить результаты.",
    ]
    for item in tasks:
        story.append(Paragraph("• " + item, bullet))
    story += [Paragraph("2. Исходные материалы", h1),
              Paragraph("Набор Adult содержит социально-демографические признаки жителей и бинарную метку дохода. "
                        "В исходной обучающей части 32 561 запись, в тестовой — 16 281 запись; всего 48 842. "
                        "Категориальное значение «?» сохранено как отдельная категория, а точка в конце меток файла adult.test удалена при загрузке.", normal),
              Paragraph('<link href="https://github.com/FeodorG/data-mining-fundamentals/tree/main/lab3" color="#1565C0"><u>https://github.com/FeodorG/data-mining-fundamentals/tree/main/lab3</u></link>', normal),
              Paragraph('Оригинал набора: <link href="https://archive.ics.uci.edu/dataset/2/adult" color="#1565C0"><u>UCI Machine Learning Repository — Adult</u></link>.', normal),
              PageBreak()]

    story += [Paragraph("3. Метод решения", h1),
              Paragraph("Дерево строится рекурсивно. Для числового признака перебираются пороги, сформированные по квантилям; для категориального — бинарные проверки равенства категории. Выбирается разбиение с наибольшим значением критерия. Ограничения глубины и минимального размера листа уменьшают переобучение и время расчёта.", normal),
              Paragraph("Information Gain: IG = H(S) - sum((|S_i|/|S|) H(S_i)).", ParagraphStyle("Formula1", parent=code, fontName="Arial")),
              Paragraph("Gain Ratio: GR = IG / SplitInfo.", ParagraphStyle("Formula2", parent=code, fontName="Arial")),
              Paragraph("Gini gain: DG = Gini(S) - sum((|S_i|/|S|) Gini(S_i)).", ParagraphStyle("Formula3", parent=code, fontName="Arial")),
              Paragraph("Положительным классом считается доход > 50K. Метрики вычисляются по TP, TN, FP и FN: accuracy — доля всех верных ответов; precision — доля верных среди предсказанных положительных; recall — доля найденных положительных; F1 — гармоническое среднее precision и recall.", normal),
              Paragraph("Параметры эксперимента", h2),
              Table([["Параметр", "Значение"], ["max_depth", "8"], ["min_samples_split", "200"],
                     ["min_samples_leaf", "80"], ["max_thresholds", "32"], ["random seed", "42"]],
                    colWidths=[7.5 * cm, 7.5 * cm], style=TableStyle([
                        ("FONT", (0, 0), (-1, -1), "Arial", 9.5), ("FONT", (0, 0), (-1, 0), "Arial-Bold", 9.5),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#246B78")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F8FA")]),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B7CBD2")),
                        ("ALIGN", (1, 1), (1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]))]

    story += [PageBreak(), Paragraph("4. Сравнение критериев", h1),
              Paragraph("Каждое дерево обучено на всех 32 561 записях adult.data и проверено на независимых 16 281 записях adult.test.", normal)]
    criterion_table = [["Критерий", "Accuracy", "Precision", "Recall", "F1"]]
    labels = {"information_gain": "Information Gain", "gain_ratio": "Gain Ratio", "gini": "Gini Index"}
    for row in data["criteria"]:
        criterion_table.append([labels[row["criterion"]], fmt(row["accuracy"]), fmt(row["precision"]), fmt(row["recall"]), fmt(row["f1"])])
    story.append(Table(criterion_table, colWidths=[5.3 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm], style=TableStyle([
        ("FONT", (0, 0), (-1, -1), "Arial", 9.5), ("FONT", (0, 0), (-1, 0), "Arial-Bold", 9.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#246B78")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F8FA")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B7CBD2")), ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ])))
    story += [Spacer(1, 8), Paragraph("Information Gain показал максимальные accuracy (0,851) и F1 (0,630). Gain Ratio немного повысил precision до 0,769, но снизил recall до 0,515. Все модели существенно хуже находят положительный класс, чем обеспечивают общую accuracy: это связано с дисбалансом классов, поскольку доля >50K составляет примерно 24%.", normal)]

    figures = [("information_gain", "Information Gain"), ("gain_ratio", "Gain Ratio"), ("gini", "Gini Index")]
    for i, (key, label) in enumerate(figures, 1):
        story += [PageBreak(), Paragraph(f"4.{i}. Дерево: {label}", h2),
                  Paragraph("На рисунке показаны первые четыре уровня. Узел содержит условие, число наблюдений, долю положительного класса и прогноз. Ветвь yes соответствует выполнению условия; цвет обозначает прогнозируемый класс.", small),
                  Spacer(1, 4), scaled_image(RESULTS / f"tree_{key}.png", 17 * cm, 12.5 * cm),
                  Paragraph(f"Рисунок {i}. Верхние уровни дерева, критерий {label}.", caption)]

    story += [PageBreak(), Paragraph("5. Влияние размера обучающей выборки", h1),
              Paragraph("Для этого эксперимента официальные части объединены, после чего выполнено стратифицированное разбиение с фиксированным seed = 42. Критерий зафиксирован: Gini Index. При каждом соотношении дерево обучалось заново.", normal)]
    split_table = [["Обучение:тест", "Обучение", "Тест", "Accuracy", "Precision", "Recall", "F1"]]
    for row in data["splits"]:
        tr = round(row["train_ratio"] * 100)
        split_table.append([f"{tr}:{100-tr}", str(row["train_size"]), str(row["test_size"]), fmt(row["accuracy"]), fmt(row["precision"]), fmt(row["recall"]), fmt(row["f1"])])
    story.append(Table(split_table, colWidths=[2.7 * cm, 2.35 * cm, 2.1 * cm, 2.45 * cm, 2.45 * cm, 2.25 * cm, 2.1 * cm], repeatRows=1, style=TableStyle([
        ("FONT", (0, 0), (-1, -1), "Arial", 8.5), ("FONT", (0, 0), (-1, 0), "Arial-Bold", 8.3),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#246B78")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F8FA")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7CBD2")), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ])))
    story += [Spacer(1, 10), scaled_image(RESULTS / "metrics_by_split.png", 17 * cm, 10.5 * cm),
              Paragraph("Рисунок 4. Зависимость показателей качества от соотношения обучающей и тестовой выборок.", caption),
              Paragraph("Увеличение доли обучения не привело к монотонному росту качества. Accuracy менялась в узком диапазоне 0,841–0,851. Наибольший F1 (0,639) наблюдался при 60:40. При 90:10 precision снизилась до 0,724. Это объясняется сочетанием трёх факторов: различным составом тестовых частей, дисбалансом классов и ограничением сложности дерева. Меньшая тестовая выборка также даёт менее устойчивую оценку. Поэтому по одному разбиению нельзя утверждать, что 60% данных принципиально лучше 90%; для строгого сравнения нужна повторная кросс-валидация.", normal),
              PageBreak()]

    story += [Paragraph("6. Выводы", h1),
              Paragraph("Разработана самостоятельная программа дерева решений с тремя критериями выбора разбиения и параметром доли обучающей выборки. Реализованы очистка Adult, стратифицированное разбиение, расчёт accuracy, precision, recall и F1, экспорт структуры дерева и визуализация результатов.", normal),
              Paragraph("На официальном разбиении лучшим по accuracy и F1 оказался Information Gain. Общая accuracy около 85% заметно выше recall положительного класса (около 52–54%), поэтому одной accuracy недостаточно для оценки модели на несбалансированных данных. Эксперимент с долями подтверждает, что увеличение обучающей части само по себе не гарантирует улучшения результата при фиксированной сложности дерева и единственном случайном разбиении.", normal),
              Paragraph("7. Состав каталога", h1),
              Paragraph("adult_tree.py — реализация дерева и CLI; run_experiments.py — полный эксперимент; data/ — исходные файлы Adult; results/ — таблицы, JSON-структуры и рисунки; requirements.txt — зависимости; README.md — инструкция запуска; report/ — данный отчёт.", normal)]

    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
