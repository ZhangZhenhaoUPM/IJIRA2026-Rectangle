# Source attribution

This release contains the project detector implementations in `code/` and shared tools in `code/common/`. Third-party authors retain their rights. Supplied license texts are included in this directory.

- RIDGE: https://github.com/Neathle/TFM . Upstream reference revision: `145b2ba18c2ec39be7a4bf578ab37708948d2c81`. See the supplied RIDGE license.
- M-LSD: https://github.com/navervision/mlsd , revision `453cafa09467d0272760578d35c1fda38e8895a5`. Copyright 2021-present NAVER Corp., Apache License v2.0. The supplied square-construction implementation is `third_party/mlsd/utils.py`.
- Ultralytics: the supplied upstream license applies to the YOLO dependency.
- AirLine/DexiNed: model definition in `third_party/dexined/model.py`, frontend and CRG adaptation in `code/ridge-airline.py`, and C++ implementation in `code/native/airline/CRGandLP_fixed.cpp`. No separate AirLine license was included with the supplied source; this release does not assign it a new license.

Dataset and pretrained-model terms remain those of their original sources.
