
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st

st.set_page_config(page_title="CV Demo Console", page_icon="🧠", layout="wide")

ASSET_DIR = Path(__file__).parent / "assets"
SAMPLE_IMAGE_PATH = ASSET_DIR / "sample_image.jpg"

try:
    import torch
    from torchvision import transforms
    from torchvision.models.segmentation import fcn_resnet50
    from torchvision.models.segmentation import FCN_ResNet50_Weights
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
    TORCHVISION_READY = True
except Exception:
    TORCHVISION_READY = False

PALETTE = np.array([
    [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189],
    [140, 86, 75], [227, 119, 194], [127, 127, 127], [188, 189, 34], [23, 190, 207]
], dtype=np.uint8)


def load_image(uploaded_file) -> Image.Image:
    if uploaded_file is not None:
        return Image.open(uploaded_file).convert("RGB")
    return Image.open(SAMPLE_IMAGE_PATH).convert("RGB")


def resize_for_display(image: Image.Image, max_width: int = 900) -> Image.Image:
    if image.width <= max_width:
        return image
    ratio = max_width / image.width
    return image.resize((max_width, int(image.height * ratio)))


def image_to_np(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def quantize_segmentation(image: Image.Image, levels: int = 4):
    arr = image_to_np(image)
    seg = ((arr // (256 // levels)) * (256 // levels) + (256 // levels) // 2).clip(0, 255).astype(np.uint8)
    return Image.fromarray(seg)


def create_detection_demo(image: Image.Image, score_threshold: float = 0.45):
    arr = image_to_np(image).astype(np.float32) / 255.0
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    sat = np.where(maxc == 0, 0, (maxc - minc) / np.maximum(maxc, 1e-6))
    mask = sat > score_threshold
    h, w = mask.shape
    candidates = []
    for y in range(0, h, 60):
        for x in range(0, w, 60):
            patch = mask[y:min(h, y + 120), x:min(w, x + 120)]
            score = float(patch.mean())
            if score > score_threshold:
                candidates.append((score, x, y, min(w, x + 120), min(h, y + 120)))

    boxes = []
    for score, x1, y1, x2, y2 in sorted(candidates, reverse=True):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if all(abs(cx - (a + c) / 2) > 80 or abs(cy - (b + d) / 2) > 80 for _, a, b, c, d in boxes):
            boxes.append((score, x1, y1, x2, y2))
        if len(boxes) >= 6:
            break

    labels = ["signage", "pedestrian", "storefront", "vehicle", "banner", "building"]
    out = image.copy()
    draw = ImageDraw.Draw(out)
    rows = []
    for idx, (score, x1, y1, x2, y2) in enumerate(boxes):
        draw.rounded_rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=4, radius=12)
        label = labels[idx % len(labels)]
        text = f"{label} {score:.2f}"
        tw = max(120, int(len(text) * 9 + 18))
        draw.rounded_rectangle([x1, max(0, y1 - 32), x1 + tw, y1], fill=(255, 80, 80), radius=8)
        draw.text((x1 + 8, max(0, y1 - 26)), text, fill="white")
        rows.append({"label": label, "score": round(score, 3), "box": [x1, y1, x2, y2]})
    return out, pd.DataFrame(rows)


def create_mask_demo(image: Image.Image):
    arr = image_to_np(image)
    mask_img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", mask_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    regions = [
        ((r > 180) & (g < 140) & (b < 150), (255, 64, 64, 100), "red instance"),
        ((g > 150) & (r < 180) & (b < 180), (64, 220, 120, 100), "green instance"),
        ((b > 140) & (r < 180), (64, 140, 255, 100), "blue instance"),
        (((r > 160) & (g > 120) & (b < 120)), (255, 210, 64, 100), "warm instance"),
    ]
    rows = []
    for rule, color, label in regions:
        ys, xs = np.where(rule)
        if len(xs) < 1500:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        draw.rounded_rectangle([x1, y1, x2, y2], fill=color, outline=color[:3] + (220,), width=3, radius=18)
        rows.append({"instance": label, "pixels": int(len(xs)), "box": [x1, y1, x2, y2]})

    combined = Image.alpha_composite(mask_img, overlay)
    return combined.convert("RGB"), pd.DataFrame(rows)


def try_real_fcn(image: Image.Image):
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights).eval()
    preprocess = weights.transforms()
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)["out"][0].argmax(0).cpu().numpy()
    color_mask = PALETTE[output % len(PALETTE)]
    return Image.fromarray(color_mask)


def try_real_faster(image: Image.Image, threshold: float = 0.5):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).eval()
    tensor = transforms.ToTensor()(image)
    with torch.no_grad():
        pred = model([tensor])[0]
    out = image.copy()
    draw = ImageDraw.Draw(out)
    rows = []
    labels = weights.meta.get("categories", [])
    for box, score, label_id in zip(pred["boxes"], pred["scores"], pred["labels"]):
        score = float(score)
        if score < threshold:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        label = labels[int(label_id)] if int(label_id) < len(labels) else str(int(label_id))
        draw.rounded_rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=4, radius=12)
        text = f"{label} {score:.2f}"
        tw = max(120, int(len(text) * 9 + 18))
        draw.rounded_rectangle([x1, max(0, y1 - 32), x1 + tw, y1], fill=(255, 80, 80), radius=8)
        draw.text((x1 + 8, max(0, y1 - 26)), text, fill="white")
        rows.append({"label": label, "score": round(score, 3), "box": [x1, y1, x2, y2]})
        if len(rows) >= 8:
            break
    return out, pd.DataFrame(rows)


def try_real_mask(image: Image.Image, threshold: float = 0.5):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights).eval()
    tensor = transforms.ToTensor()(image)
    with torch.no_grad():
        pred = model([tensor])[0]
    base = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    labels = weights.meta.get("categories", [])
    rows = []
    masks = pred["masks"].detach().cpu().numpy()
    for idx, (box, score, label_id, mask) in enumerate(zip(pred["boxes"], pred["scores"], pred["labels"], masks)):
        score = float(score)
        if score < threshold:
            continue
        binary = mask[0] > threshold
        ys, xs = np.where(binary)
        if len(xs) == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        color = tuple(int(v) for v in PALETTE[idx % len(PALETTE)]) + (100,)
        draw.rounded_rectangle([x1, y1, x2, y2], fill=color, outline=color[:3] + (220,), width=3, radius=18)
        label = labels[int(label_id)] if int(label_id) < len(labels) else str(int(label_id))
        rows.append({"instance": label, "score": round(score, 3), "pixels": int(binary.sum())})
        if len(rows) >= 6:
            break
    combined = Image.alpha_composite(base, overlay)
    return combined.convert("RGB"), pd.DataFrame(rows)


def main():
    st.title("Vibe Coding 计算机视觉交互 Demo")
    st.caption("提交版默认优先保证网页可部署，因此内置 Demo 模式；如果运行环境支持 torchvision，可切换到真实模型分支。")

    with st.sidebar:
        st.header("参数设置")
        uploaded = st.file_uploader("上传一张 JPG / PNG 图片", type=["jpg", "jpeg", "png"])
        score_threshold = st.slider("阈值 / 置信度", 0.20, 0.90, 0.55, 0.05)
        use_real_models = st.checkbox(
            "尝试调用真实 torchvision 模型",
            value=False,
            disabled=not TORCHVISION_READY,
            help="当前环境不支持时会自动回退到 Demo 模式。"
        )
        if TORCHVISION_READY:
            st.success("检测到可选的 torchvision 环境")
        else:
            st.info("当前运行 Demo 模式，部署更稳，更适合课程提交")
        st.markdown("---")
        st.markdown("**内置说明**")
        st.write("1. 不上传图片时默认使用项目自带样例图。")
        st.write("2. FCN 展示语义分割效果。")
        st.write("3. Faster R-CNN 展示目标框结果。")
        st.write("4. Mask R-CNN 展示实例掩膜叠加结果。")

    image = resize_for_display(load_image(uploaded))

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("输入图片")
        st.image(image, use_container_width=True)
    with c2:
        st.subheader("项目定位")
        st.write(
            "本项目面向课程展示：用一个可交互的 Streamlit 网页统一演示 FCN、Faster R-CNN、Mask R-CNN 三类视觉任务，"
            "同时给出 R-CNN / Fast R-CNN 与现代两阶段方法的结构对比。"
        )
        st.write(
            "为了兼顾部署稳定性与可演示性，网页默认运行轻量 Demo 分支；如果你本地环境支持 PyTorch + torchvision，"
            "勾选侧边栏开关后可尝试真实预训练模型。"
        )

    tab1, tab2, tab3, tab4 = st.tabs(["FCN 语义分割", "Faster R-CNN 检测", "Mask R-CNN 实例分割", "方法对比"])

    with tab1:
        st.subheader("FCN 语义分割示例")
        if use_real_models and TORCHVISION_READY:
            try:
                result = try_real_fcn(image)
                mode_name = "真实 torchvision FCN"
            except Exception as e:
                st.warning(f"真实模型运行失败，已切回 Demo 模式：{e}")
                result = quantize_segmentation(image)
                mode_name = "Demo 模式"
        else:
            result = quantize_segmentation(image)
            mode_name = "Demo 模式"
        st.caption(f"当前模式：{mode_name}")
        a, b = st.columns(2)
        a.image(image, caption="原图", use_container_width=True)
        b.image(result, caption="分割结果", use_container_width=True)
        st.write("这里采用像素级区域着色来直观说明语义分割的核心思想：同一类区域会被映射为同一种颜色。")

    with tab2:
        st.subheader("Faster R-CNN 目标检测示例")
        if use_real_models and TORCHVISION_READY:
            try:
                det_img, det_df = try_real_faster(image, threshold=score_threshold)
                mode_name = "真实 torchvision Faster R-CNN"
            except Exception as e:
                st.warning(f"真实模型运行失败，已切回 Demo 模式：{e}")
                det_img, det_df = create_detection_demo(image, score_threshold=score_threshold)
                mode_name = "Demo 模式"
        else:
            det_img, det_df = create_detection_demo(image, score_threshold=score_threshold)
            mode_name = "Demo 模式"
        st.caption(f"当前模式：{mode_name}")
        a, b = st.columns([1.25, 1])
        a.image(det_img, caption="检测结果", use_container_width=True)
        b.dataframe(det_df, use_container_width=True, hide_index=True)
        st.write("网页报告中同步解释 R-CNN → Fast R-CNN → Faster R-CNN 的演进逻辑，便于老师看懂模型脉络。")

    with tab3:
        st.subheader("Mask R-CNN 实例分割示例")
        if use_real_models and TORCHVISION_READY:
            try:
                mask_img, mask_df = try_real_mask(image, threshold=score_threshold)
                mode_name = "真实 torchvision Mask R-CNN"
            except Exception as e:
                st.warning(f"真实模型运行失败，已切回 Demo 模式：{e}")
                mask_img, mask_df = create_mask_demo(image)
                mode_name = "Demo 模式"
        else:
            mask_img, mask_df = create_mask_demo(image)
            mode_name = "Demo 模式"
        st.caption(f"当前模式：{mode_name}")
        a, b = st.columns([1.25, 1])
        a.image(mask_img, caption="实例分割结果", use_container_width=True)
        b.dataframe(mask_df, use_container_width=True, hide_index=True)
        st.write("Mask R-CNN 在检测框基础上进一步输出实例级掩膜，因此展示效果通常最强，也最适合课程汇报截图。")

    with tab4:
        st.subheader("R-CNN / Fast R-CNN / Faster R-CNN / FCN / Mask R-CNN 对比")
        compare_df = pd.DataFrame([
            ["FCN", "语义分割", "像素级类别图", "结构清晰，适合课程展示", "不区分同类实例", "推荐保留"],
            ["R-CNN", "目标检测", "候选框 + 分类", "便于讲经典两阶段思路", "速度慢，不利于网页演示", "仅在报告讲解"],
            ["Fast R-CNN", "目标检测", "共享特征图 + 分类回归", "比 R-CNN 更高效", "仍依赖外部候选框", "仅在报告讲解"],
            ["Faster R-CNN", "目标检测", "检测框 + 置信度", "经典强基线，结构完整", "推理较重", "网页主展示"],
            ["Mask R-CNN", "实例分割", "检测框 + 实例掩膜", "展示效果最好", "资源占用更高", "网页主展示"],
        ], columns=["方法", "任务粒度", "输出", "优点", "局限", "提交建议"])
        st.dataframe(compare_df, use_container_width=True, hide_index=True)
        st.info("如果你最后要交一个最稳的可访问 URL，建议保留当前轻量版 requirements.txt；如果追求真实模型效果，再自行补装 torch / torchvision。")

    st.markdown("---")
    st.markdown("**样例图片说明**：默认样例图已内置在 `assets/sample_image.jpg`，便于直接部署演示。")


if __name__ == "__main__":
    main()
