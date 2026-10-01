# Native code / C++源码

| Path | Function |
|---|---|
| `graph/graph_support.cpp` | Measure observed line support between graph vertices |
| `graph/cycles.cpp` | AirLine quadrilateral search and geometric checks |
| `graph/cycles_fld.cpp` | FLD quadrilateral search and geometric checks |
| `graph/centroid/cycles.cpp` | AirLine candidate search with centroid-based corner ordering |
| `graph/centroid/cycles_fld.cpp` | FLD candidate search with centroid-based corner ordering |
| `airline/CRGandLP_fixed.cpp` | AirLine CRG line extraction with the supplied compatibility fixes |

`code/common/runtime.py` copies the graph sources to a temporary build directory and compiles them using `g++ -O2 -shared -std=c++17 -fPIC -ffp-contract=off`. The two corner-ordering implementations remain separate because they define different candidate construction behavior. FLD/AirLine load the resulting shared libraries through `ctypes`.

Fresh AirLine frontend inference compiles the CRG Python extension from `airline/CRGandLP_fixed.cpp`, using the Python and PyTorch/pybind11 headers. The cached frontend path does not invoke that extension. All binaries are built for the local Linux environment and removed with the temporary directory at process exit.

Parameters are read from `parameters/`; JSON files in the temporary build directory are generated from those settings. Source attribution is in `licenses/SOURCES.md`.

本目录按图计算和AirLine前端分组，包含实际使用的六份C++源码。编译入口在 `code/common/runtime.py` 和 `code/ridge-airline.py`。参数统一从 `parameters/` 读取，临时JSON及编译产物自动生成，无需手动维护。
