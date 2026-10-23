from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from adult_tree import DecisionTree, Node, classification_metrics, load_adult, stratified_split, xy


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"accuracy": "#264653", "precision": "#e76f51", "recall": "#2a9d8f", "f1": "#e9c46a"}

WINDOWS_FONTS = Path(r"C:\Windows\Fonts")


# --------------------------------------------------------------------------- #
# Шрифты
# --------------------------------------------------------------------------- #
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = WINDOWS_FONTS / ("arialbd.ttf" if bold else "arial.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# График метрик от доли обучения
# --------------------------------------------------------------------------- #
def plot_metrics(rows: list[dict], output: Path) -> None:
    width, height = 1500, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 185, 120, 1420, 730
    draw.text((width // 2, 42), "Качество классификации в зависимости от доли обучения",
              fill="#1d3557", font=font(32, True), anchor="ma")
    for value in np.arange(0.55, 0.91, 0.05):
        y = bottom - (value - 0.55) / 0.35 * (bottom - top)
        draw.line((left, y, right, y), fill="#d9e2ec", width=2)
        draw.text((left - 18, y), f"{value:.2f}", fill="#334e68", font=font(22), anchor="rm")
    draw.line((left, top, left, bottom), fill="#334e68", width=3)
    draw.line((left, bottom, right, bottom), fill="#334e68", width=3)

    ratios = [r["train_ratio"] for r in rows]
    xs = [left + i * (right - left) / (len(ratios) - 1) for i in range(len(ratios))]
    for x, ratio in zip(xs, ratios):
        draw.line((x, bottom, x, bottom + 10), fill="#334e68", width=3)
        train_percent = round(ratio * 100)
        draw.text((x, bottom + 24), f"{train_percent}:{100 - train_percent}",
                  fill="#334e68", font=font(22), anchor="ma")

    for metric, color in COLORS.items():
        points = [(x, bottom - (row[metric] - 0.55) / 0.35 * (bottom - top))
                  for x, row in zip(xs, rows)]
        draw.line(points, fill=color, width=6, joint="curve")
        for x, y in points:
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline="white", width=2)

    draw.text(((left + right) // 2, 825), "Соотношение обучающей и тестовой выборок",
              fill="#334e68", font=font(24, True), anchor="mm")
    draw.text((92, (top + bottom) // 2), "Значение",
              fill="#334e68", font=font(24, True), anchor="mm")

    legend_x = 245
    for metric, color in COLORS.items():
        draw.line((legend_x, 92, legend_x + 45, 92), fill=color, width=6)
        draw.text((legend_x + 58, 92), metric, fill="#243b53", font=font(21), anchor="lm")
        legend_x += 245

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


# --------------------------------------------------------------------------- #
# Отрисовка дерева решений на Pillow (без Graphviz)
# --------------------------------------------------------------------------- #
def _collect_nodes(root: Node, max_depth: int) -> list[Node]:
    """Обход в глубину: собираем узлы, которые будут нарисованы."""
    result: list[Node] = []

    def walk(node: Node) -> None:
        result.append(node)
        if node.is_leaf or node.depth >= max_depth:
            return
        walk(node.left)
        walk(node.right)

    walk(root)
    return result


def _assign_x(root: Node, max_depth: int) -> tuple[dict[int, float], int]:
    """Возвращает координаты X для каждого узла (в единицах «листовых слотов»)
    и общее число листовых слотов. Узел стоит посередине между детьми."""
    xs: dict[int, float] = {}
    counter = {"next": 0}

    def walk(node: Node) -> float:
        if node.is_leaf or node.depth >= max_depth:
            x = counter["next"]
            counter["next"] += 1
            xs[id(node)] = x
            return x
        lx = walk(node.left)
        rx = walk(node.right)
        x = (lx + rx) / 2.0
        xs[id(node)] = x
        return x

    walk(root)
    return xs, max(counter["next"], 1)


def _node_caption(node: Node, max_depth: int) -> str:
    if node.is_leaf or node.depth >= max_depth:
        first = "лист" if node.is_leaf else "поддерево скрыто"
    elif node.threshold is not None:
        first = f"{node.feature} <= {node.threshold:.3g}"
    else:
        first = f"{node.feature} = {node.category}"
    probability = node.positive / node.samples
    cls = ">50K" if node.prediction else "<=50K"
    return f"{first}\nsamples = {node.samples}\nP(>50K) = {probability:.3f}\nclass = {cls}"


def draw_tree(tree: DecisionTree, output: Path, display_depth: int = 4) -> None:
    """Рендер верхних уровней дерева решений в PNG без внешних утилит."""
    if tree.root is None:
        raise RuntimeError("fit must be called first")

    nodes = _collect_nodes(tree.root, display_depth)
    xs_units, total_slots = _assign_x(tree.root, display_depth)

    # Размеры узла и отступы
    node_w, node_h = 260, 130
    margin_x, margin_y = 40, 40
    header_h = 70
    h_gap, v_gap = 24, 120

    used_slots = total_slots
    width = margin_x * 2 + used_slots * node_w + max(used_slots - 1, 0) * h_gap
    depth_count = max(n.depth for n in nodes) + 1
    height = header_h + margin_y * 2 + depth_count * node_h + (depth_count - 1) * v_gap
    # Не даём изображению стать слишком большим — верхние уровни и так широкие
    width = max(width, 1200)
    height = max(height, 700)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    title = f"Дерево решений (первые {display_depth} уровня) — критерий: {tree.criterion}"
    draw.text((width // 2, 28), title, fill="#17324D", font=font(24, True), anchor="ma")

    def x_pixel(unit: float) -> float:
        return margin_x + node_w / 2 + unit * (node_w + h_gap)

    def y_pixel(depth: int) -> float:
        return header_h + margin_y + node_h / 2 + depth * (node_h + v_gap)

    positions: dict[int, tuple[float, float]] = {}
    for node in nodes:
        positions[id(node)] = (x_pixel(xs_units[id(node)]), y_pixel(node.depth))

    # Сначала линии, потом прямоугольники — так узлы перекрывают концы линий
    for node in nodes:
        if node.is_leaf or node.depth >= display_depth:
            continue
        px, py = positions[id(node)]
        for child, label in ((node.left, "yes"), (node.right, "no")):
            if id(child) not in positions:
                continue
            cx, cy = positions[id(child)]
            draw.line(
                (px, py + node_h / 2, cx, cy - node_h / 2),
                fill="#7b8794", width=3,
            )
            # подпись ветви посередине
            mx, my = (px + cx) / 2, (py + node_h / 2 + cy - node_h / 2) / 2
            draw.text((mx + 6, my - 6), label,
                      fill="#52606d", font=font(16, True), anchor="lm")

    body_font = font(15)
    for node in nodes:
        px, py = positions[id(node)]
        fill = "#f7b267" if node.prediction else "#74a9cf"
        draw.rounded_rectangle(
            (px - node_w / 2, py - node_h / 2, px + node_w / 2, py + node_h / 2),
            radius=12, fill=fill, outline="#334e68", width=3,
        )
        draw.multiline_text(
            (px, py), _node_caption(node, display_depth),
            fill="#102a43", font=body_font, anchor="mm", align="center",
            spacing=4,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


# --------------------------------------------------------------------------- #
# Основной сценарий
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "adult.data.txt")
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "adult.test.txt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--display-depth", type=int, default=4)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    official_train = load_adult(args.train)
    official_test = load_adult(args.test)
    x_train, y_train = xy(official_train)
    x_test, y_test = xy(official_test)

    # Эксперимент 1: 100% обучающей выборки, три критерия
    criterion_rows = []
    for criterion in ("information_gain", "gain_ratio", "gini"):
        print(f"Training {criterion} on official split...", flush=True)
        tree = DecisionTree(criterion=criterion, max_depth=args.max_depth).fit(x_train, y_train)
        metrics = classification_metrics(y_test, tree.predict(x_test))
        criterion_rows.append({
            "criterion": criterion,
            "train_size": len(official_train),
            "test_size": len(official_test),
            **metrics,
        })
        (out / f"tree_{criterion}.json").write_text(
            json.dumps(tree.to_dict(), indent=2), encoding="utf-8"
        )
        print(f"  drawing tree_{criterion}.png ...", flush=True)
        draw_tree(tree, out / f"tree_{criterion}.png", display_depth=args.display_depth)

    with (out / "criterion_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=criterion_rows[0].keys())
        writer.writeheader()
        writer.writerows(criterion_rows)

    # Эксперимент 2: Gini, разные доли обучения
    import pandas as pd
    all_data = pd.concat([official_train, official_test], ignore_index=True)
    ratio_rows = []
    for ratio in (0.6, 0.7, 0.8, 0.9):
        print(f"Training gini with ratio {ratio:.1f}...", flush=True)
        train, test = stratified_split(all_data, ratio, seed=42)
        xt, yt = xy(train)
        xv, yv = xy(test)
        tree = DecisionTree(criterion="gini", max_depth=args.max_depth).fit(xt, yt)
        metrics = classification_metrics(yv, tree.predict(xv))
        ratio_rows.append({
            "train_ratio": ratio,
            "test_ratio": 1 - ratio,
            "train_size": len(train),
            "test_size": len(test),
            **metrics,
        })

    with (out / "split_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ratio_rows[0].keys())
        writer.writeheader()
        writer.writerows(ratio_rows)

    plot_metrics(ratio_rows, out / "metrics_by_split.png")

    summary = {"criteria": criterion_rows, "splits": ratio_rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()