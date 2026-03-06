 # 疲劳驾驶实时检测（YOLOv11 + ResNet-50）

项目基于 PyTorch 实现 YOLOv11 + ResNet-50 双阶段实时疲劳驾驶检测系统，覆盖多种驾驶场景（白天/夜间、裸眼/佩戴眼镜），在 NTHU 数据集上的分类准确率达 98%，检测 mAP@0.5 达 92%，实时检测帧率达 30 FPS。

---

## 目录（快速导航）
- [概述](#概述)
- [特性](#特性)
- [项目结构](#项目结构)
- [环境与依赖](#环境与依赖)
- [快速开始](#快速开始)
- [数据准备](#数据准备)
- [训练](#训练)
- [推理 / 集成测试](#推理--集成测试)
- [结果与评估指标](#结果与评估指标)
- [文档与实验记录](#文档与实验记录)
- [联系方式](#联系方式)

---

## 概述
疲劳驾驶是导致重大交通事故的主要原因之一，传统检测方式难以满足实时性要求。本项目采用双阶段架构：YOLOv11 负责人脸检测，ResNet-50 负责疲劳/非疲劳分类，实现检测与分类任务解耦；提供训练、推理、前端实时演示与服务化能力。

---

## 特性
- 双阶段架构：YOLOv11（检测） + ResNet-50（分类）
- 支持训练/微调、模型导出（ONNX/torchscript 可选）
- 支持摄像头实时推理（OpenCV 演示），实时叠加检测框、标签与报警
- 多场景覆盖（白天/夜间、裸眼/佩戴眼镜）
- 支持完整训练流水线与日志、实验记录管理

---

## 项目结构
项目根目录示例（与你给出的结构保持一致）：

```
fatigue_detection/
│
├── dataset/                     # 数据集
│   ├── resnet_format/           # ResNet分类数据
│   │   ├── train/
│   │   │     ├──fatigue
│   │   │     └──non_fatigue
│   │   ├── val/
│   │   │     ├──fatigue
│   │   │     └──non_fatigue
│   │   └── test/
│   │   │     ├──fatigue
│   │   │     └──non_fatigue
│   └── yolo_format/             # YOLO检测数据
│       ├── images/
│       │     ├──test
│       │     ├──train
│       │     └──val
│       └── labels/
│             ├──test
│             ├──train
│             └──val
│
├── models/                  # 模型结构
│   ├── yolo/                # YOLOv11人脸检测
│   │   ├── yolo_model.py
│   │   └── yolo_config.yaml
│   │
│   ├── resnet/              # ResNet疲劳分类
│   │   ├── resnet50.py
│   │   └── classifier.py
│   │
│   └── build_model.py       # 模型构建统一入口
│
├── training/                # 训练代码
│   ├── train_yolo.py
│   ├── train_resnet.py
│   └── trainer.py
│
├── inference/               # 推理与部署
│   ├── detect_face.py       # YOLO检测
│   ├── classify_fatigue.py  # ResNet分类
│   └── pipeline.py          # 完整推理流程
│
├── preprocessing/           # 数据预处理
│   ├── video_to_frames.py   # 视频抽帧
│   ├── crop_faces.py        # 根据bbox裁剪人脸
│   └── augmentations.py     # 数据增强
│
├── utils/                   # 工具函数
│   ├── metrics.py           # Accuracy/F1/AUC
│   ├── visualization.py     # 可视化
│   ├── logger.py
│   └── config.py
│
├── configs/                 # 配置文件
│   ├── yolo_train.yaml
│   ├── resnet_train.yaml
│   └── inference.yaml
│
├── checkpoints/             # 训练权重
│   ├── yolo/
│   └── resnet/
│
├── experiments/             # 实验结果
│   ├── logs/
│   ├── plots/
│   └── reports/
│
├── demo/                    # 实时检测Demo
│   └── webcam_demo.py
│
├── requirements.txt         # 依赖
├── README.md                # 项目说明
└── main.py                  # 程序入口
```

---

## 环境与依赖
推荐使用 Python 3.8+ 与 CUDA（若有 GPU）。主要依赖示例（详见 `requirements.txt`）：
- torch / torchvision
- opencv-python
- pyyaml
- numpy
- tqdm
- matplotlib
- （可选）onnx, onnxruntime

安装依赖：
```bash
pip install -r requirements.txt
```

---

## 快速开始

1. 准备数据（参考下方“数据准备”）
2. 配置训练/推理参数：编辑 `configs/resnet_train.yaml`、`configs/yolo_train.yaml`、`configs/inference.yaml`
3. 训练检测模型（示例）：
```bash
python inference/training/train_yolo.py --config configs/yolo_train.yaml
```
4. 训练分类模型（示例）：
```bash
python inference/training/train_resnet.py --config configs/resnet_train.yaml
```
5. 运行摄像头演示：
```bash
python demo/webcam_demo.py --config configs/inference.yaml --weights_yolo checkpoints/yolo/best.pt --weights_resnet checkpoints/resnet/best.pth
```
（示例参数请根据实际脚本调整）

---

## 数据准备
- `dataset/resnet_format/`：分类模型数据，按 `train/val/test` 划分；每类一个子文件夹（fatigue / non_fatigue）。
- `dataset/yolo_format/`：检测模型数据，采用 images + labels（COCO-like 或 YOLO txt 格式）。
- 建议把 NTHU 数据集与自采集数据合并后按场景划分（白天/夜间、裸眼/佩戴眼镜），并保持 train/val/test 的不重叠。

常用预处理脚本位于：
- `inference/preprocessing/video_to_frames.py`
- `inference/preprocessing/crop_faces.py`
- `inference/preprocessing/augmentations.py`

---

## 训练细节（要点）
- 检测：基于 COCO 预训练权重微调 YOLOv11，使用数据增强并调优置信度阈值，mAP@0.5 可达 92%。
- 分类：基于 ImageNet 预训练权重微调 ResNet-50，引入数据增强、早停与 StepLR 学习率衰减，分类准确率达 98%。
- 推荐使用实验管理（logs / plots / experiments）保存超参、训练曲线与 checkpoint。

---

## 推理与系统集成
- 推理脚本：`inference/pipeline.py`（组合检测 + 分类，支持视频/摄像头输入）
- 单独模块：`inference/detect_face.py`、`inference/classify_fatigue.py`
- 实时演示：`demo/webcam_demo.py`，基于 OpenCV 叠加检测框、分类标签、置信度与报警提示，帧率约 30 FPS（视硬件而定）
- 可选：导出 ONNX 以便在不同平台/设备上使用加速推理（参考 `models/*/export`）

---

## 结果与评估指标
- 检测：mAP@0.5 = 92%
- 分类：准确率 = 98%
- 实时性能：检测帧率 ≈ 30 FPS（OpenCV 摄像头演示）

---

## 文档与实验记录
详见 `docs/`（技术报告、实验记录、评估报告与训练日志）。

---

## 联系方式
- 联系：GitHub - xinyi-0509
