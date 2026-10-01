# Dataset annotations / 数据集标注

Each image subdirectory has one annotation file:

| File | Images | Quadrilaterals | Object semantics |
|---|---:|---:|---|
| `images/robotathome80/annotations.json` | 80 | 146 | Original categories, occlusions and scene conditions |
| `images/isaacsim50/annotations.json` | 50 | 76 | Not available |
| `images/smartdoc90/annotations.json` | 90 | 90 | Not available |

Each JSON maps an image filename to its annotation record. `quads` contains four-corner polygons with normalized `(x, y)` coordinates. Multiply x by `width` and y by `height` for pixel coordinates.

Only RobotHome80 has a `categories` field: `categories[i]` is the original semantic label for `quads[i]`. Original category spellings and metadata are preserved. IsaacSim50 and SmartDoc90 contain geometry annotations. SmartDoc's `document_group` is publisher grouping metadata, not an object semantic label.

`manifest.json` is the combined evaluation index for all 220 images. Every polygon and its order exactly match the corresponding per-dataset `quads`; image files and evaluation geometry are unchanged.

三个子数据集各有一个`annotations.json`，就在各自的图像目录内。只有RobotHome80含语义标签，`categories`与`quads`中的矩形一一对应，并保留遮挡及场景条件。IsaacSim50和SmartDoc90只有几何标注；SmartDoc的`document_group`单独保留为分组信息。类别拼写按原始文件保留，未自行更改。
