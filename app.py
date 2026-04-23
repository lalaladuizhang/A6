from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageColor, ImageDraw, ImageFont

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

APP_DIR = Path(__file__).resolve().parent
ASSET_DIR = APP_DIR / 'assets'
SAMPLE_IMAGE = ASSET_DIR / 'sample_image.jpg'

PALETTE = [
    '#e63946', '#f4a261', '#e9c46a', '#2a9d8f', '#457b9d',
    '#5e60ce', '#7209b7', '#43aa8b', '#f72585', '#4d908e'
]

COCO_LABELS = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)


def cv_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def resize_keep_ratio(img: Image.Image, max_side: int = 960) -> Image.Image:
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def make_label_legend(class_names: List[str]) -> pd.DataFrame:
    return pd.DataFrame({'类别': class_names, '说明': ['演示识别结果'] * len(class_names)})


def kmeans_segmentation(pil_img: Image.Image, k: int = 6, alpha: float = 0.45) -> Tuple[Image.Image, Dict[str, float]]:
    img = pil_to_cv(resize_keep_ratio(pil_img, 1000))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = rgb.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    centers = np.uint8(centers)

    color_palette = np.array([ImageColor.getrgb(PALETTE[i % len(PALETTE)]) for i in range(k)], dtype=np.uint8)
    mask_color = color_palette[labels].reshape(rgb.shape)
    blended = cv2.addWeighted(rgb, 1 - alpha, mask_color, alpha, 0)

    counts = np.bincount(labels, minlength=k)
    ratios = counts / counts.sum()
    stats = {f'区域{i+1}': float(round(r, 4)) for i, r in enumerate(ratios)}
    return Image.fromarray(blended), stats


def contour_boxes(pil_img: Image.Image, max_boxes: int = 6) -> Tuple[Image.Image, List[Dict[str, float]]]:
    img = pil_to_cv(resize_keep_ratio(pil_img, 1000))
    draw = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 80, 180)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    candidates = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < 0.01 * w * h or area > 0.75 * w * h:
            continue
        score = area / (w * h)
        candidates.append((score, x, y, bw, bh))

    candidates.sort(reverse=True)
    results = []
    for idx, (_, x, y, bw, bh) in enumerate(candidates[:max_boxes], start=1):
        cv2.rectangle(draw, (x, y), (x + bw, y + bh), (56, 189, 248), 3)
        cv2.putText(draw, f'proposal_{idx}', (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (56, 189, 248), 2)
        results.append({
            'label': f'proposal_{idx}',
            'x1': int(x), 'y1': int(y), 'x2': int(x + bw), 'y2': int(y + bh),
            'score': round(float((bw * bh) / (w * h)), 3),
        })

    return cv_to_pil(draw), results


def grabcut_instance(pil_img: Image.Image) -> Tuple[Image.Image, Dict[str, float]]:
    img = pil_to_cv(resize_keep_ratio(pil_img, 1000))
    h, w = img.shape[:2]
    rect = (int(0.08 * w), int(0.08 * h), int(0.84 * w), int(0.84 * h))
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    foreground = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    overlay = img.copy()
    red = np.zeros_like(img)
    red[:, :, 2] = 255
    overlay = np.where(foreground[:, :, None] == 1, cv2.addWeighted(img, 0.55, red, 0.45, 0), img)

    contours, _ = cv2.findContours((foreground * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    area_ratio = float(foreground.sum() / (h * w))
    contour_count = len(contours)
    stats = {'mask_area_ratio': round(area_ratio, 4), 'instance_count_demo': contour_count}
    return cv_to_pil(overlay), stats


def try_real_torchvision_inference(pil_img: Image.Image, task: str):
    try:
        import torch
        from torchvision import transforms
        from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
        from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    except Exception as e:
        return None, f'当前环境无法启用 torchvision 真实推理：{e}'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pil_img = resize_keep_ratio(pil_img, 900)
    x = transforms.ToTensor()(pil_img).to(device)

    if task == 'FCN 语义分割':
        weights = FCN_ResNet50_Weights.DEFAULT
        model = fcn_resnet50(weights=weights).to(device).eval()
        with torch.no_grad():
            out = model(x.unsqueeze(0))['out'][0]
        mask = out.argmax(0).cpu().numpy().astype(np.uint8)
        palette = np.array([ImageColor.getrgb(PALETTE[i % len(PALETTE)]) for i in range(21)], dtype=np.uint8)
        color_mask = palette[mask % len(palette)]
        base = np.array(pil_img.convert('RGB'))
        merged = cv2.addWeighted(base, 0.55, color_mask, 0.45, 0)
        return Image.fromarray(merged), None

    if task == 'Faster R-CNN 目标检测':
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()
        with torch.no_grad():
            pred = model([x])[0]
        base = pil_to_cv(pil_img)
        keep = pred['scores'].cpu().numpy() > 0.5
        for box, label, score in zip(pred['boxes'][keep], pred['labels'][keep], pred['scores'][keep]):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            name = COCO_LABELS[int(label)]
            cv2.rectangle(base, (x1, y1), (x2, y2), (56, 189, 248), 2)
            cv2.putText(base, f'{name}:{float(score):.2f}', (x1, max(24, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)
        return cv_to_pil(base), None

    if task == 'Mask R-CNN 实例分割':
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn(weights=weights).to(device).eval()
        with torch.no_grad():
            pred = model([x])[0]
        base = np.array(pil_img.convert('RGB'))
        keep = pred['scores'].cpu().numpy() > 0.5
        for idx, mask in enumerate(pred['masks'][keep]):
            m = (mask[0].cpu().numpy() > 0.5)
            color = np.array(ImageColor.getrgb(PALETTE[idx % len(PALETTE)]), dtype=np.uint8)
            base[m] = (0.55 * base[m] + 0.45 * color).astype(np.uint8)
        return Image.fromarray(base), None

    return None, '未知任务'


def run_demo_pipeline(pil_img: Image.Image, task: str, mode: str):
    start = time.perf_counter()
    if mode == '真实推理（需 torchvision 与权重）':
        output, err = try_real_torchvision_inference(pil_img, task)
        if output is not None:
            elapsed = time.perf_counter() - start
            return output, {'mode': 'real', 'time_sec': round(elapsed, 3)}
        mode = '演示模式（默认可运行）'
        fallback_message = err
    else:
        fallback_message = None

    if task == 'FCN 语义分割':
        output, stats = kmeans_segmentation(pil_img)
    elif task == 'Faster R-CNN 目标检测':
        output, stats = contour_boxes(pil_img)
    else:
        output, stats = grabcut_instance(pil_img)

    elapsed = time.perf_counter() - start
    meta = {'mode': 'demo', 'time_sec': round(elapsed, 3), 'stats': stats}
    if fallback_message:
        meta['warning'] = fallback_message
    return output, meta


def comparison_table() -> pd.DataFrame:
    return pd.DataFrame([
        ['R-CNN', '候选框 + CNN 分类', '目标检测', '慢，分阶段', '概念清晰，但推理最慢'],
        ['Fast R-CNN', '共享卷积特征 + RoI Pooling', '目标检测', '中等', '训练/推理都优于 R-CNN'],
        ['Faster R-CNN', '引入 RPN 自动生成候选框', '目标检测', '较快', '经典两阶段检测器'],
        ['FCN', '全卷积逐像素预测', '语义分割', '较快', '适合像素级类别分配'],
        ['Mask R-CNN', 'Faster R-CNN + mask 分支', '实例分割', '较慢', '同时输出框与像素掩码'],
    ], columns=['方法', '核心思想', '输出形式', '速度感知', '说明'])


def app_main() -> None:
    st.set_page_config(page_title='Vibe Coding CV Demo', layout='wide')
    st.title('Vibe Coding：检测与分割交互演示')
    st.caption('支持 FCN 语义分割、Faster R-CNN 目标检测、Mask R-CNN 实例分割，并附带方法对比页面。')

    with st.sidebar:
        st.header('控制面板')
        task = st.selectbox('选择任务', ['FCN 语义分割', 'Faster R-CNN 目标检测', 'Mask R-CNN 实例分割', '方法性能对比'])
        mode = st.radio('运行模式', ['演示模式（默认可运行）', '真实推理（需 torchvision 与权重）'])
        uploaded = st.file_uploader('上传图片', type=['jpg', 'jpeg', 'png'])
        use_sample = st.checkbox('使用内置样例图', value=True if uploaded is None else False)

    if uploaded is not None:
        pil_img = Image.open(uploaded).convert('RGB')
    elif use_sample and SAMPLE_IMAGE.exists():
        pil_img = Image.open(SAMPLE_IMAGE).convert('RGB')
    else:
        st.warning('请上传图片，或启用内置样例图。')
        return

    pil_img = resize_keep_ratio(pil_img)

    if task == '方法性能对比':
        left, right = st.columns([1, 1])
        with left:
            st.image(pil_img, caption='输入图片', use_container_width=True)
        with right:
            st.dataframe(comparison_table(), use_container_width=True, hide_index=True)
            st.info('建议课堂演示时：先展示 Faster R-CNN 与 Mask R-CNN 的区别，再补充 FCN 的语义分割定位。')
        return

    result, meta = run_demo_pipeline(pil_img, task, mode)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('原图')
        st.image(pil_img, use_container_width=True)
    with col2:
        st.subheader('处理结果')
        st.image(result, use_container_width=True)

    st.markdown('### 运行信息')
    st.write({'任务': task, '模式': meta['mode'], '耗时(秒)': meta['time_sec']})
    if 'warning' in meta:
        st.warning(meta['warning'])
    if 'stats' in meta and meta['stats']:
        if isinstance(meta['stats'], list):
            st.dataframe(pd.DataFrame(meta['stats']), use_container_width=True, hide_index=True)
        else:
            st.json(meta['stats'])

    with st.expander('实验说明'):
        st.write(
            '本项目默认提供演示模式，以确保在课堂、无 GPU 或无预训练权重的环境里也能直接运行。'
            '若本地安装了兼容版本的 torchvision 并允许下载权重，可切换到真实推理模式。'
        )


if __name__ == '__main__':
    if st is None:
        raise SystemExit('请先安装 streamlit：pip install streamlit')
    app_main()
