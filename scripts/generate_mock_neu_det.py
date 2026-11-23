"""生成符合 NEU-DET 标准目录与 VOC XML 格式的模拟金属表面缺陷数据集。

特点：
1. 生成 200x200 灰度金属表面底图，叠加 6 类具有清晰视觉差异的缺陷纹理：
   - crazing（裂纹）：网状/树枝状细暗裂纹
   - inclusion（夹杂）：不规则深色长条/椭圆夹杂颗粒
   - patches（斑块）：大块不规则明暗斑块与晕染
   - pitted_surface（麻点）：密集分布的针孔状小凹坑
   - rolled-in_scale（压入氧化皮）：沿轧制方向成片分布的鱼鳞状灰暗压痕
   - scratches（划痕）：细长、笔直的高亮/暗色划伤线条
2. 同步生成标准 PASCAL VOC XML 标注文件（包含缺陷边界框 bndbox），
   既可用于 PyTorch ResNet18 分类训练，也可直接导出 YOLO 目标检测格式。
3. 目录布局遵循 NEU-DET 检测版标准（train / validation，每类默认 60 张用于快速演练，
   可通过 --per-class 300 生成与官方同等规模的数据集）。
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vision_agent.config import DEFECT_CLASSES


def _make_metal_background(rng: np.random.Generator, size: int = 200) -> np.ndarray:
    """生成带轻微轧制条纹与颗粒噪声的灰度金属底图。"""
    base_level = rng.uniform(118.0, 148.0)
    # 水平轧制条纹
    row_pattern = rng.normal(0.0, 4.5, size=(size, 1))
    col_grad = np.linspace(-8.0, 8.0, size)[None, :] * rng.uniform(-1.0, 1.0)
    noise = rng.normal(0.0, 6.5, size=(size, size))
    img = base_level + row_pattern + col_grad + noise
    return np.clip(img, 30, 230).astype(np.uint8)


def _draw_crazing(img: Image.Image, py_rng: random.Random) -> list[tuple[int, int, int, int]]:
    """绘制网状/分叉龟裂纹。"""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    bboxes: list[tuple[int, int, int, int]] = []
    num_clusters = py_rng.randint(2, 3)

    for _ in range(num_clusters):
        cx = py_rng.randint(35, w - 35)
        cy = py_rng.randint(35, h - 35)
        xs, ys = [cx], [cy]
        # 从中心向多个角度发散锯齿折线构成网状裂纹
        for _branch in range(py_rng.randint(6, 10)):
            x, y = float(cx), float(cy)
            angle = py_rng.uniform(0, 2 * math.pi)
            for _step in range(py_rng.randint(5, 9)):
                step_len = py_rng.uniform(6, 12)
                angle += py_rng.uniform(-0.7, 0.7)
                nx = max(4.0, min(w - 5.0, x + math.cos(angle) * step_len))
                ny = max(4.0, min(h - 5.0, y + math.sin(angle) * step_len))
                shade = py_rng.randint(25, 65)
                draw.line([(x, y), (nx, ny)], fill=shade, width=py_rng.choice([1, 2]))
                x, y = nx, ny
                xs.append(int(x))
                ys.append(int(y))
        bboxes.append((max(1, min(xs) - 4), max(1, min(ys) - 4), min(w - 1, max(xs) + 4), min(h - 1, max(ys) + 4)))
    return bboxes


def _draw_inclusion(img: Image.Image, py_rng: random.Random) -> list[tuple[int, int, int, int]]:
    """绘制深色孤立长条/椭圆形非金属夹杂。"""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    bboxes: list[tuple[int, int, int, int]] = []
    for _ in range(py_rng.randint(2, 4)):
        cx = py_rng.randint(25, w - 25)
        cy = py_rng.randint(25, h - 25)
        rx = py_rng.randint(4, 10)
        ry = py_rng.randint(14, 34)
        if py_rng.random() < 0.35:
            rx, ry = ry, rx
        x1, y1 = max(2, cx - rx), max(2, cy - ry)
        x2, y2 = min(w - 2, cx + rx), min(h - 2, cy + ry)
        # 外围浅暗晕
        draw.ellipse([x1 - 2, y1 - 2, x2 + 2, y2 + 2], fill=py_rng.randint(65, 95))
        # 深黑核心夹杂物
        draw.ellipse([x1, y1, x2, y2], fill=py_rng.randint(15, 45))
        bboxes.append((x1, y1, x2, y2))
    return bboxes


def _draw_patches(img: Image.Image, py_rng: random.Random) -> list[tuple[int, int, int, int]]:
    """绘制大面积不规则高对比度斑块。"""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    bboxes: list[tuple[int, int, int, int]] = []
    for _ in range(py_rng.randint(2, 4)):
        cx = py_rng.randint(35, w - 35)
        cy = py_rng.randint(35, h - 35)
        num_pts = py_rng.randint(6, 9)
        pts = []
        for i in range(num_pts):
            ang = (2 * math.pi * i) / num_pts + py_rng.uniform(-0.25, 0.25)
            rad = py_rng.uniform(22, 46)
            px = int(max(3, min(w - 3, cx + math.cos(ang) * rad)))
            py = int(max(3, min(h - 3, cy + math.sin(ang) * rad)))
            pts.append((px, py))
        shade = py_rng.choice([py_rng.randint(28, 62), py_rng.randint(190, 232)])
        draw.polygon(pts, fill=shade)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bboxes.append((min(xs), min(ys), max(xs), max(ys)))
    return bboxes


def _draw_pitted_surface(img: Image.Image, py_rng: random.Random) -> list[tuple[int, int, int, int]]:
    """绘制密集分布的针孔状麻点凹坑。"""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    bboxes: list[tuple[int, int, int, int]] = []
    for _ in range(py_rng.randint(1, 2)):
        x1 = py_rng.randint(15, 85)
        y1 = py_rng.randint(15, 85)
        x2 = min(w - 10, x1 + py_rng.randint(75, 115))
        y2 = min(h - 10, y1 + py_rng.randint(75, 115))
        for _dot in range(py_rng.randint(45, 95)):
            dx = py_rng.randint(x1 + 3, x2 - 3)
            dy = py_rng.randint(y1 + 3, y2 - 3)
            r = py_rng.randint(1, 3)
            draw.ellipse([dx - r, dy - r, dx + r, dy + r], fill=py_rng.randint(20, 60))
            if py_rng.random() < 0.4:
                draw.point((dx + r, dy - r), fill=py_rng.randint(180, 220))
        bboxes.append((x1, y1, x2, y2))
    return bboxes


def _draw_rolled_in_scale(img: Image.Image, py_rng: random.Random) -> list[tuple[int, int, int, int]]:
    """绘制沿水平轧制方向分布的鱼鳞状压入氧化皮。"""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    bboxes: list[tuple[int, int, int, int]] = []
    for _ in range(py_rng.randint(2, 3)):
        bx = py_rng.randint(20, w - 75)
        by = py_rng.randint(20, h - 60)
        bw = py_rng.randint(45, 75)
        bh = py_rng.randint(28, 52)
        # 鱼鳞状弧形叠片
        for _scale in range(py_rng.randint(8, 15)):
            sx = py_rng.randint(bx, bx + bw - 14)
            sy = py_rng.randint(by, by + bh - 10)
            sw = py_rng.randint(10, 20)
            sh = py_rng.randint(6, 12)
            draw.arc([sx, sy, sx + sw, sy + sh], start=0, end=180, fill=py_rng.randint(30, 65), width=2)
            draw.chord([sx + 1, sy + 1, sx + sw - 1, sy + sh - 1], start=15, end=165, fill=py_rng.randint(70, 100))
        bboxes.append((bx, by, min(w - 1, bx + bw), min(h - 1, by + bh)))
    return bboxes


def _draw_scratches(img: Image.Image, py_rng: random.Random) -> list[tuple[int, int, int, int]]:
    """绘制细长笔直的高亮或暗色划痕。"""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    bboxes: list[tuple[int, int, int, int]] = []
    for _ in range(py_rng.randint(2, 4)):
        vertical_like = py_rng.random() < 0.75
        if vertical_like:
            x1 = py_rng.randint(18, w - 18)
            y1 = py_rng.randint(5, 45)
            x2 = max(5, min(w - 5, x1 + py_rng.randint(-18, 18)))
            y2 = py_rng.randint(h - 55, h - 5)
        else:
            x1 = py_rng.randint(5, 45)
            y1 = py_rng.randint(18, h - 18)
            x2 = py_rng.randint(w - 55, w - 5)
            y2 = max(5, min(h - 5, y1 + py_rng.randint(-18, 18)))
        shade = py_rng.choice([py_rng.randint(215, 250), py_rng.randint(20, 48)])
        width = py_rng.choice([1, 2])
        draw.line([(x1, y1), (x2, y2)], fill=shade, width=width)
        # 紧邻边缘阴影增强划痕反光立体感
        draw.line([(x1 + 2, y1), (x2 + 2, y2)], fill=max(10, shade - 70) if shade > 150 else min(245, shade + 75), width=1)
        bboxes.append((max(1, min(x1, x2) - 4), max(1, min(y1, y2) - 4), min(w - 1, max(x1, x2) + 4), min(h - 1, max(y1, y2) + 4)))
    return bboxes


_DEFECT_DRAWERS = {
    "crazing": _draw_crazing,
    "inclusion": _draw_inclusion,
    "patches": _draw_patches,
    "pitted_surface": _draw_pitted_surface,
    "rolled-in_scale": _draw_rolled_in_scale,
    "scratches": _draw_scratches,
}


def _write_voc_xml(
    xml_path: Path,
    filename: str,
    width: int,
    height: int,
    defect_class: str,
    bboxes: list[tuple[int, int, int, int]],
) -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "NEU-DET"
    ET.SubElement(root, "filename").text = filename

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "1"

    for xmin, ymin, xmax, ymax in bboxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = defect_class
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(max(1, xmin))
        ET.SubElement(bnd, "ymin").text = str(max(1, ymin))
        ET.SubElement(bnd, "xmax").text = str(min(width, max(xmin + 2, xmax)))
        ET.SubElement(bnd, "ymax").text = str(min(height, max(ymin + 2, ymax)))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=False)


def generate_mock_neu_det_dataset(
    output_dir: str | Path = "data/neu_det",
    per_class: int = 60,
    val_ratio: float = 0.2,
    image_size: int = 200,
    seed: int = 42,
) -> dict[str, int]:
    """生成完整 NEU-DET 模拟数据集（200x200 灰度 JPG + VOC XML 标注）。"""
    out_root = Path(output_dir)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    val_per_class = max(1, int(round(per_class * val_ratio)))
    train_per_class = max(1, per_class - val_per_class)

    total_train = 0
    total_val = 0

    for cls_name in DEFECT_CLASSES:
        drawer = _DEFECT_DRAWERS[cls_name]
        for split_name, count, start_idx in (
            ("train", train_per_class, 1),
            ("validation", val_per_class, train_per_class + 1),
        ):
            img_dir = out_root / split_name / "images" / cls_name
            ann_dir = out_root / split_name / "annotations"
            img_dir.mkdir(parents=True, exist_ok=True)
            ann_dir.mkdir(parents=True, exist_ok=True)

            for offset in range(count):
                idx = start_idx + offset
                bg = _make_metal_background(rng, size=image_size)
                img = Image.fromarray(bg, mode="L")
                bboxes = drawer(img, py_rng)
                if cls_name in ("patches", "rolled-in_scale"):
                    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))

                stem = f"{cls_name}_{idx}"
                img_file = f"{stem}.jpg"
                img.save(img_dir / img_file, quality=92)
                _write_voc_xml(
                    ann_dir / f"{stem}.xml",
                    filename=img_file,
                    width=image_size,
                    height=image_size,
                    defect_class=cls_name,
                    bboxes=bboxes,
                )

                if split_name == "train":
                    total_train += 1
                else:
                    total_val += 1

    return {
        "total_images": total_train + total_val,
        "train_images": total_train,
        "val_images": total_val,
        "classes": len(DEFECT_CLASSES),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 NEU-DET 格式模拟金属表面缺陷图像与 VOC XML 标注")
    parser.add_argument("--output", default="data/neu_det", help="输出目录（默认 data/neu_det）")
    parser.add_argument("--per-class", type=int, default=60, help="每个缺陷类别生成的图片数（真实标准为 300）")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例（默认 0.2）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    stats = generate_mock_neu_det_dataset(
        output_dir=args.output,
        per_class=args.per_class,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(
        f"[OK] 已生成 NEU-DET 模拟数据集：{args.output} | "
        f"共 {stats['total_images']} 张（训练 {stats['train_images']} / 验证 {stats['val_images']}，{stats['classes']} 类）"
    )


if __name__ == "__main__":
    main()
