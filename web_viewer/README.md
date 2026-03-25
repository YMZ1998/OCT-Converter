# Viewer 文档与接口说明

本文档整理当前仓库内与查看器相关的启动入口、目录结构、CLI 参数、HTTP 接口和维护边界。

## 1. 当前保留的启动入口

顶层只保留真正给用户使用的启动入口：

- `web_viewer/oct_web_viewer.py`：OCT Web Viewer 启动入口
- `web_viewer/fa_web_viewer.py`：FA Web Viewer 启动入口
- `scripts/fa_qt_viewer.py`：统一 FA Qt Viewer 启动入口

其余实现代码已经按模态拆分到子目录中，避免顶层继续堆积兼容壳文件。

## 2. 目录结构

```text
web_viewer/
├─ oct_web_viewer.py         # OCT Web 启动入口
├─ fa_web_viewer.py          # FA Web 启动入口
├─ oct/
│  ├─ app.py                 # OCT CLI 参数与服务启动
│  ├─ server.py              # OCT HTTP 路由
│  ├─ state.py               # OCT 数据加载、状态组织、PNG/JSON 输出
│  ├─ vendor_modes.py        # OCT 厂商模式与输入校验
│  ├─ viewer.html            # OCT 前端页面
│  ├─ shiwei_loader.py       # 视微目录式 DICOM 解析
│  ├─ tupai_loader.py        # Tupai 目录式 DICOM 解析
│  └─ zeiss_loader.py        # Zeiss DICOM/OCT 辅助解析
└─ fa/
   ├─ app.py                 # FA CLI 参数与服务启动
   ├─ server.py              # FA HTTP 路由
   ├─ state.py               # FA 数据加载、状态组织、PNG 输出、最近路径持久化
   ├─ viewer.html            # FA 前端页面
   └─ viewer.py              # FA 本地页面封装

scripts/
├─ fa_qt_viewer.py           # 统一 FA Qt 启动入口
└─ fa/
   ├─ fa_qt_viewer.py        # 统一 FA Qt 主实现，也被 Web FA 复用
   ├─ topcon_fa_qt.py        # Topcon FA 解析
   ├─ zeiss_fa_qt.py         # Zeiss FA 解析
   └─ hdb_fa_qt.py           # HDB / Heidelberg FA 解析
```

## 3. OCT Web Viewer

### 3.1 用途

用于本地快速查看 OCT 数据，包括：

- B-scan 切片浏览
- projection / fundus 图查看
- 层分割轮廓叠加
- 扫描框 / 定位线可视化
- 本地路径加载、文件选择和单文件上传

### 3.2 支持输入

支持扩展名：

- `.fds`
- `.fda`
- `.e2e`
- `.img`
- `.oct`
- `.OCT`
- `.dcm`
- `.dicom`

支持的 `vendor` 模式：

- `auto`
- `heidelberg`
- `topcon`
- `shiwei`
- `tupai`

说明：

- `auto` 会按路径和文件类型自动识别。
- `shiwei`、`tupai` 属于目录式 DICOM 数据集，优先使用路径加载，不适合浏览器单文件上传。
- Zeiss `.img`、Bioptigen `.oct`、Optovue `.OCT`、通用 DICOM 主要走标准 reader 路径，不单独暴露专用 `vendor` 值。

### 3.3 启动方式

```bash
python web_viewer/oct_web_viewer.py
python web_viewer/oct_web_viewer.py /path/to/file.e2e
python web_viewer/oct_web_viewer.py /path/to/file.img --img-rows 1024 --img-cols 512 --img-interlaced
```

常用参数：

- `path`：可选，启动时立即加载的 OCT 文件或目录路径
- `--host`：监听地址，当前默认值是 `192.168.0.90`
- `--port`：监听端口，默认 `8765`
- `--img-rows` / `--img-cols`：Zeiss `.img` 的尺寸参数
- `--img-interlaced`：Zeiss `.img` 按交错格式读取

说明：

- 如果本机没有 `192.168.0.90` 这个地址，建议显式传 `--host 127.0.0.1`。
- 当前实现默认不自动打开浏览器；启动后直接访问终端打印的 URL 即可。

### 3.4 对外 Python 接口

主要入口与职责：

- `web_viewer/oct/app.py`
  - `parse_args()`：解析 CLI 参数
  - `main()`：创建 `ThreadingHTTPServer` 并启动服务
- `web_viewer/oct/server.py`
  - `build_handler(state, html_path)`：构建请求处理器类
- `web_viewer/oct/state.py`
  - `ViewerState.load(path, vendor_mode)`：按路径加载数据
  - `ViewerState.pick_and_load(vendor_mode)`：弹出文件选择器并加载
  - `ViewerState.load_uploaded_file(filename, fileobj, vendor_mode)`：处理浏览器上传
  - `ViewerState.build_state_payload()`：生成前端状态 JSON
  - `ViewerState.get_slice_png(...)`：输出 B-scan PNG
  - `ViewerState.get_projection_png(index)`：输出 projection PNG
  - `ViewerState.get_contours(volume_index, slice_index)`：输出轮廓 JSON
  - `ViewerState.get_fundus_png(index)`：输出 fundus PNG
  - `ViewerState.get_volume_fundus_view_png(index)`：输出带叠加底图的 fundus 视图

## 4. FA Web Viewer

### 4.1 用途

统一查看 Topcon、Zeiss、HDB / Heidelberg 的 FA 数据，当前界面是网页形式，布局参考 Zeiss 风格。

### 4.2 支持输入

支持以下数据源：

- Topcon FA 文件夹
- Zeiss DICOM 文件或目录
- HDB / Heidelberg `.E2E` 文件

支持的 `vendor` 模式：

- `auto`
- `topcon`
- `zeiss`
- `hdb`

### 4.3 启动方式

```bash
python web_viewer/fa_web_viewer.py
python web_viewer/fa_web_viewer.py /path/to/fa_dataset --vendor auto
python web_viewer/fa_web_viewer.py /path/to/file.e2e --vendor hdb
```

常用参数：

- `path`：可选，启动时立即加载的 FA 数据路径
- `--vendor`：启动厂商模式，支持 `auto/topcon/zeiss/hdb`
- `--host`：监听地址，默认 `127.0.0.1`
- `--port`：监听端口，默认 `8766`
- `--no-browser`：不自动打开浏览器

### 4.4 页面能力

当前页面已实现：

- Zeiss 风格的网页界面
- 路径输入、选文件、选目录
- 最近路径记忆
- `Group` 分组筛选
- `眼别` 分组筛选
- `Proofsheet` 隐藏开关
- 帧表格固定显示 10 行
- `Frames` 面板固定在主图右侧
- 主图视窗可手动拖拽调整大小
- 对比度 / 亮度调节
- 播放、滑条切帧、元数据查看

最近路径同时有前后端两层持久化：

- 浏览器本地：`localStorage`
- 后端文件：`~/.oct_converter/fa_web_viewer_state.json`

后端最多保留最近 `8` 条路径。

### 4.5 对外 Python 接口

主要入口与职责：

- `web_viewer/fa/app.py`
  - `parse_args()`：解析 CLI 参数
  - `main()`：创建 `ThreadingHTTPServer` 并启动服务
- `web_viewer/fa/server.py`
  - `build_handler(state, html_path)`：构建请求处理器类
- `web_viewer/fa/state.py`
  - `FAViewerState.load(path, vendor_mode)`：按路径加载 FA 数据
  - `FAViewerState.pick_and_load_file(vendor_mode)`：弹出文件选择器并加载
  - `FAViewerState.pick_and_load_directory(vendor_mode)`：弹出目录选择器并加载
  - `FAViewerState.build_state_payload()`：生成前端状态 JSON
  - `FAViewerState.get_frame_png(index, contrast_percent, brightness_offset)`：输出帧 PNG

Web FA 复用了 `scripts/fa/fa_qt_viewer.py` 中的统一解析接口，主要包括：

- `load_unified_dataset(...)`
- `build_frame_metadata_text(...)`
- `build_summary_text(...)`
- `UnifiedFADataset`
- `UnifiedFAFrame`

## 5. 统一 FA Qt Viewer

Qt 版本仍然保留，适合需要桌面窗口和本地交互的场景。

启动方式：

```bash
python scripts/fa_qt_viewer.py
python scripts/fa_qt_viewer.py /path/to/fa_dataset --vendor auto
python scripts/fa_qt_viewer.py /path/to/file.e2e --vendor hdb
python scripts/fa_qt_viewer.py /path/to/topcon_folder --vendor topcon
python scripts/fa_qt_viewer.py /path/to/zeiss_series --vendor zeiss
```

参数说明：

- `input_path`：Topcon 文件夹、Zeiss 文件夹 / DICOM 文件、或 HDB `.E2E`
- `--vendor`：`auto/topcon/zeiss/hdb`
- `--dump`：只解析并输出帧信息，不打开 Qt 窗口

实现关系：

- `scripts/fa_qt_viewer.py` 是启动入口
- `scripts/fa/fa_qt_viewer.py` 是主实现
- `scripts/fa/topcon_fa_qt.py`、`scripts/fa/zeiss_fa_qt.py`、`scripts/fa/hdb_fa_qt.py` 分别负责厂商解析

## 6. HTTP 接口说明

### 6.1 OCT Web 接口

基础规则：

- 页面接口返回 `HTML`
- 状态接口返回 `JSON`
- 图像接口返回 `PNG`
- 失败时通常返回 `400`，格式为 `{"error": "..."}`
- 未知路由返回 `404`，格式为 `{"error": "Unknown route: ..."}`

接口列表：

- `GET /`
- `GET /index.html`
  - 返回查看器页面

- `GET /api/state`
  - 返回当前状态 payload

- `GET /api/load?path=<path>&vendor=<mode>`
  - 从本地路径加载数据
  - `vendor` 支持 `auto|heidelberg|topcon|shiwei|tupai`

- `GET /api/pick-file?vendor=<mode>`
  - 打开原生文件选择器并加载
  - 返回值会额外带 `cancelled: true|false`

- `POST /api/upload?vendor=<mode>`
  - 使用 `multipart/form-data`
  - 上传字段名为 `file`
  - 仅适合单文件场景

- `GET /api/volume/<index>/slice/<slice>.png?contrast=<int>&brightness=<int>`
  - 返回指定 B-scan PNG

- `GET /api/volume/<index>/projection.png`
  - 返回指定 volume 的 projection PNG

- `GET /api/volume/<index>/contours/<slice>.json`
  - 返回指定切片的轮廓 JSON

- `GET /api/fundus/<index>.png`
  - 返回 fundus PNG

- `GET /api/volume/<index>/fundus-view.png`
  - 返回带叠加视图的 fundus PNG

### 6.2 FA Web 接口

基础规则与 OCT 一致：页面返回 `HTML`，状态返回 `JSON`，图像返回 `PNG`，错误返回 `{"error": "..."}`。

接口列表：

- `GET /`
- `GET /index.html`
  - 返回查看器页面

- `GET /api/state`
  - 返回当前状态 payload

- `GET /api/load?path=<path>&vendor=<mode>`
  - 从本地路径加载 FA 数据
  - `vendor` 支持 `auto|topcon|zeiss|hdb`

- `GET /api/pick-file?vendor=<mode>`
  - 打开原生文件选择器并加载
  - 返回值会额外带 `cancelled: true|false`

- `GET /api/pick-directory?vendor=<mode>`
  - 打开原生目录选择器并加载
  - 返回值会额外带 `cancelled: true|false`

- `GET /api/frame/<index>.png?contrast=<int>&brightness=<int>`
  - 返回指定 FA 帧的 PNG

## 7. 状态 payload 说明

### 7.1 OCT `/api/state`

未加载时：

```json
{
  "loaded": false,
  "sourcePath": "",
  "reader": "",
  "volumes": [],
  "fundusImages": [],
  "supportedExtensions": [".fds", ".fda", ".e2e", ".img", ".oct", ".OCT", ".dcm", ".dicom"]
}
```

加载后核心字段：

- `loaded`：是否已加载
- `sourcePath`：当前数据源路径
- `sourceKind`：来源类型，当前主要为 `path` 或 `upload`
- `recentPath`：前端可复用的最近路径
- `reader`：实际 reader / 数据集名称
- `volumes`：volume 描述数组
- `fundusImages`：fundus 描述数组
- `supportedExtensions`：支持扩展名列表

### 7.2 FA `/api/state`

未加载时：

```json
{
  "loaded": false,
  "sourcePath": "",
  "vendorMode": "auto",
  "supportedVendors": [
    {"value": "auto", "label": "自动"},
    {"value": "topcon", "label": "Topcon"},
    {"value": "zeiss", "label": "Zeiss"},
    {"value": "hdb", "label": "HDB"}
  ],
  "recentPaths": [],
  "groups": [],
  "lateralityOptions": []
}
```

加载后核心字段：

- `loaded`：是否已加载
- `sourcePath`：当前输入路径
- `vendorMode`：当前厂商模式
- `supportedVendors`：可选厂商模式列表
- `recentPaths`：最近路径
- `dataset`：数据集概览
- `groups`：可筛选分组
- `lateralityOptions`：可筛选眼别
- `frames`：帧列表

其中 `dataset` 目前包含：

- `inputPath`
- `vendor`
- `vendorLabel`
- `patientName`
- `patientId`
- `sex`
- `birthDate`
- `examDate`
- `laterality`
- `studyCode`
- `deviceModel`
- `frameCount`
- `groupSummary`
- `timeRange`
- `summaryText`

`frames` 中常用字段包括：

- `originalIndex`
- `vendor`
- `filename`
- `sourcePath`
- `groupKey` / `groupLabel`
- `lateralityKey` / `lateralityLabel`
- `label`
- `sourceDetail`
- `acquisitionDisplay`
- `elapsedDisplay`
- `width` / `height`
- `isProofsheet`
- `metadataText`

## 8. 维护建议

常见修改入口：

- 改启动参数：`web_viewer/oct/app.py`、`web_viewer/fa/app.py`
- 改 HTTP 路由：`web_viewer/oct/server.py`、`web_viewer/fa/server.py`
- 改状态组织 / 输出字段：`web_viewer/oct/state.py`、`web_viewer/fa/state.py`
- 改 FA 解析逻辑：`scripts/fa/fa_qt_viewer.py` 及厂商解析脚本
- 改网页布局与交互：`web_viewer/oct/viewer.html`、`web_viewer/fa/viewer.html`

建议验证命令：

```bash
python web_viewer/oct_web_viewer.py --help
python web_viewer/fa_web_viewer.py --help
python scripts/fa_qt_viewer.py --help
```

如果改了 Python 文件，也建议至少执行一次：

```bash
python -m py_compile web_viewer/oct_web_viewer.py web_viewer/fa_web_viewer.py scripts/fa_qt_viewer.py
```
