# Dataset / 数据集

This repository includes all 220 evaluation images and 312 annotated quadrilaterals. Each subdataset has one `annotations.json` beside its images:

- `images/robotathome80/`: 80 images, 146 quadrilaterals and original object semantic labels.
- `images/isaacsim50/`: 50 images and 76 quadrilaterals.
- `images/smartdoc90/`: 90 images and 90 document quadrilaterals.

See [README_annotations.md](README_annotations.md) for field definitions. `manifest.json` fixes image order and checks correspondence with the published benchmark.

From the repository root, run `python scripts/verify_bundle.py --datasets` to verify all image and annotation checksums.

本目录已包含全部220张评估图片和312个矩形标注，无需另行下载。三个子数据集各有一份 `annotations.json`；只有RobotHome80提供对象语义标签。标注格式详见上述说明。
