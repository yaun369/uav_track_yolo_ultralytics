# UAV Track with Ultralytics (YOLO Track)

基于 Ultralytics `track` 模式实现的视频无人机追踪脚本，适配你已经训练好的 `yolo26n` 系列 `.pt` 模型。

## TODO（先做清单）

- [x] 设计命令行参数（模型路径、视频路径、tracker、阈值、设备等）
- [x] 编写可直接运行的 Python 脚本，调用 `YOLO(...).track(...)`
- [x] 加入输入校验与运行摘要输出（帧数、轨迹 ID 统计）
- [x] 增加工程化基础配置（`requirements.txt`、`.gitignore`、自定义 tracker 配置）
- [x] 提供本地 macOS 运行步骤与命令示例

## 目录结构

```text
.
├── configs/
│   └── drone_bytetrack.yaml
├── scripts/
│   └── track_drone.py
├── requirements.txt
└── README.md
```

## 1) 环境准备（macOS）

> 推荐 Python 3.10+

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) 执行无人机追踪

### 基础用法（默认 BoT-SORT）

```bash
python scripts/track_drone.py \
  --model /path/to/your/best.pt \
  --source /path/to/video.mp4 \
  --show \
  --save-video
```

### 使用 ByteTrack（官方或自定义配置）

```bash
# 官方默认配置
python scripts/track_drone.py \
  --model /path/to/your/best.pt \
  --source /path/to/video.mp4 \
  --tracker bytetrack.yaml \
  --conf 0.2 --iou 0.5 \
  --show --save-video

# 使用项目内针对无人机场景的配置
python scripts/track_drone.py \
  --model /path/to/your/best.pt \
  --source /path/to/video.mp4 \
  --tracker configs/drone_bytetrack.yaml \
  --show --save-video
```

## 3) Apple Silicon / Mac 设备建议

如果你是 M 系列芯片，可以尝试启用 MPS：

```bash
python scripts/track_drone.py \
  --model /path/to/your/best.pt \
  --source /path/to/video.mp4 \
  --device mps \
  --show --save-video
```

若 MPS 环境不稳定，改用 `--device cpu` 可提升兼容性。

## 4) 参数说明

- `--model`：训练好的 `.pt` 模型路径（必填）
- `--source`：输入视频路径（必填，支持 mp4/mov/avi/mkv/m4v）
- `--tracker`：`botsort.yaml` / `bytetrack.yaml` / 自定义 yaml
- `--conf`：置信度阈值（默认 `0.25`）
- `--iou`：NMS IoU 阈值（默认 `0.45`）
- `--classes`：仅追踪某些类别，如 `--classes 0`
- `--show`：弹窗显示实时追踪画面
- `--save-video`：保存带框视频到 `runs/track/...`
- `--device`：`cpu` / `mps` / `cuda:0`

## 5) 结果输出

脚本结束后会打印：

- 处理帧数
- 追踪到的唯一目标 ID 数量
- （启用 `--save-video` 时）输出目录提示

