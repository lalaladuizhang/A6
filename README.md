# Vibe Coding CV Demo

这是一个面向课程提交的 Streamlit 交互式 Demo，用一个网页统一展示：

- FCN 语义分割示例
- Faster R-CNN 目标检测示例
- Mask R-CNN 实例分割示例
- R-CNN / Fast R-CNN / Faster R-CNN 的方法对比

## 你真正需要上传到 GitHub 的文件

把下面这些内容放到仓库根目录即可：

- `app.py`
- `requirements.txt`
- `README.md`
- `assets/sample_image.jpg`
- `.streamlit/config.toml`

## 不建议上传的内容

- 实验报告 Word / PDF
- 渲染截图、临时缓存
- `__pycache__`
- 本地日志文件

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud 部署

1. 新建一个 GitHub 仓库。
2. 把上述 5 个必要内容上传到仓库根目录。
3. 登录 Streamlit Community Cloud。
4. 点击 **Create app**。
5. 选择你的 GitHub 仓库。
6. Main file path 填 `app.py`。
7. 点击 Deploy，等待生成 `https://xxx.streamlit.app` 链接。

## 说明

提交版默认优先保证网页稳定可部署，因此 requirements 使用轻量依赖。
如果你后续想切换为真实 torchvision 模型，可在 `app.py` 中保留的分支上继续扩展，并自行安装 `torch` / `torchvision`。
