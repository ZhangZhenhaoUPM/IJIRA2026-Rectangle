# 矩形检测复现包 1.0

本版本包含RIDGE-FLD、RIDGE-AirLine、RIDGE-ROI、RIDGE-YOLO、YOLO-BR和M-LSD六种方法，评估数据集为RobotHome80、IsaacSim50和SmartDoc90。四个RIDGE变体在同一数据集共用8项几何参数，保留不同后端和各方法的候选筛选设置。

## 目录

- `dataset/`：已包含三个评估子集的220张图片及对应标注，无需另行下载。
- `code/`：各方法的主要算法和命令行入口。
- `code/common/`：FLD与AirLine共用的图计算工具，以及C++编译、临时配置管理、数据集标注读取和共享参数解析。
- `third_party/`：DexiNed模型定义、官方M-LSD工具，各保留一份。
- `code/native/`：图计算与AirLine线段提取的C++源码；编译产物写入临时目录。
- `parameters/`：六个方法各一份最终配置，加一份`shared_geometry.json`保存八项共享几何参数。详见[参数说明](parameters/README.md)。
- `configs/`：下载链接及数据、模型、前端缓存的校验清单。
- `reference/`：论文结果表CSV及18组逐图预测JSON，用于重算五项指标、核对重新推理的结果。
- `evaluation/`、`scripts/`：评估和算法复现入口。
- `licenses/`：来源归属及许可证。

## 准备与运行

环境安装命令见[README.md](README.md)。完整推理使用Linux、Python 3.12、g++及NVIDIA GPU；M-LSD使用独立Python环境。

克隆本仓库，或点击 **Code > Download ZIP** 后解压。数据集已包含在仓库的 `dataset/` 中：

```text
dataset/
  manifest.json
  images/
    robotathome80/   # 80张图片 + annotations.json
    isaacsim50/      # 50张图片 + annotations.json
    smartdoc90/      # 90张图片 + annotations.json
```

从项目根目录运行：

1. `python scripts/verify_bundle.py --datasets`：检查代码、参数、全部图片和标注。
2. `python scripts/evaluate_results.py --output outputs/reference_evaluation`：从参考预测重算18行五指标；此步骤不需要模型。

完整推理需要从 [Release v1.0](https://github.com/ZhangZhenhaoUPM/IJIRA2026-Rectangle/releases/tag/v1.0) 下载 [models.zip](https://github.com/ZhangZhenhaoUPM/IJIRA2026-Rectangle/releases/download/v1.0/models.zip)，可选下载 [frontend_cache.zip](https://github.com/ZhangZhenhaoUPM/IJIRA2026-Rectangle/releases/download/v1.0/frontend_cache.zip)。将这两个压缩包解压到项目外的 `../ridge_assets/`，得到 `../ridge_assets/models/` 和 `../ridge_assets/frontend_cache/`。下载命令和环境安装命令见英文README。

```bash
python scripts/prepare_assets.py --assets ../ridge_assets
python scripts/reproduce.py --output outputs/inference --device cuda:0 --mlsd-python .venv-mlsd/bin/python
```

不下载前端缓存时，推理命令需加 `--fresh-airline`，使用AirLine神经网络重新提取线段。下载链接及压缩包SHA256保存在 `configs/downloads.json`。

输出目录必须是新目录。论文统一几何参数通过复现脚本应用；各方法文件已包含最终筛选设置。直接运行底层方法文件不会安装共享几何参数的运行钩子，请使用上述复现命令。

## 指标与验证

18组推理的坐标已核对，15组已导出分数的结果也逐项一致，五指标与结果表按四位小数一致。验证时重新编译了`code/native/`中的C++源码，AirLine的220张图使用完整神经前端推理。当前源码清单、导入路径检查和验证范围见`VALIDATION.json`。

Accuracy和Specificity为256×256掩码上的像素级指标；Precision、Recall和F1为IoU阈值0.5下的实例级指标。数据集参与了参数选择，结果属于开发集比较。IsaacSim50上的YOLO-BR使用图像级十折OOF权重，其训练协议与RIDGE-YOLO的固定权重不同。本包覆盖算法推理和定量评估。

## 数据集标注文件

仓库的`dataset/images/robotathome80/`、`isaacsim50/`、`smartdoc90/`各有一个`annotations.json`。只有RobotHome80含原始对象语义，`categories`与矩形逐项对应；另外两个数据集提供几何标注。SmartDoc的文档分组仅作为分组信息保留。`code/common/datasets.py`直接读取各子数据集的标注文件，以`dataset/manifest.json`校验图像顺序和几何标注。推理入口仅接收图像信息，不传入标注。
