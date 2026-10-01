# Parameters / 参数

There are seven JSON files: one per method and one for the geometry shared by the four RIDGE variants. Each file contains settings for RobotHome80, IsaacSim50 and SmartDoc90.

| File | Contents |
|---|---|
| `shared_geometry.json` | Eight shared RIDGE geometry thresholds, grouped by dataset |
| `ridge-fld.json` | Selected FLD frontend, candidate construction, ranking and output limits |
| `ridge-airline.json` | Selected AirLine frontend, candidate construction, ranking and output limits |
| `ridge-roi.json` | Contour proposals, ROI processing, selected candidate filtering and frontend protocols |
| `ridge-yolo.json` | Detector checkpoints, proposal thresholds and method-specific settings |
| `yolo-br.json` | Detector and boundary selection settings; fixed IsaacSim50 OOF image assignments |
| `mlsd.json` | Official M-LSD square construction settings and output limits of 6/6/1 |

Use `python scripts/reproduce.py --output outputs/inference --device cuda:0 --mlsd-python .venv-mlsd/bin/python` from the project root. It reads the method files and applies `shared_geometry.json` to all four RIDGE backends. The shared geometry is fixed within each dataset; backends and candidate selection remain method-specific. Direct engine calls do not install these shared-geometry hooks.

FLD and AirLine store the final selection settings under `datasets` and backend protocol templates under `pool_protocols`. RIDGE-ROI stores its final settings under `datasets`, per-dataset line frontend settings under `frontends`, and SmartDoc preprocessing under `smartdoc_frontend`. An entry such as `{"$shared": "angle_threshold"}` reads that value from `shared_geometry.json` for the active dataset. Protocol group names identify candidate pools; they do not select a second numerical geometry profile. The reconstruction protocols are resolved before each dataset run. The frontend line-cache extension of 1.25 is retained as a coordinate convention; reconstruction uses the shared elongation factor.

Pixel thresholds are measured in each backend's working coordinates. RIDGE-YOLO scales point distances internally by two; area fractions refer to the reconstruction image or crop. Model checksums are in `configs/models_sha256.json`. Reference predictions and metrics are in `reference/` and are only used to check outputs.

## 中文说明

目录中共七份 JSON：六个方法各一份，四个 RIDGE 共用一份 `shared_geometry.json`。各方法文件内按三个数据集组织设置。FLD/AirLine 的 `datasets` 保存最终筛选参数，`pool_protocols` 保存构建候选所需的模板。ROI 的 `datasets` 保存最终设置，`frontends` 和 `smartdoc_frontend` 保存实际使用的前端设置；`$shared` 表示读取共享参数文件中的对应数值。

论文复现通过 `scripts/reproduce.py` 运行。代码直接读取这些最终设置，再按数据集应用八项共享几何参数。各方法的后端、前端分辨率和候选筛选方式保持各自定义。运行时会将预测与 `reference/` 中的参考结果比较。

数据、模型和前端缓存的校验清单位于 `configs/`。修改数值后，应重新推理和评估；固定参考结果的核对会报告差异。
