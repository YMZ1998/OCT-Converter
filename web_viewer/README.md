# OCT Web Viewer 说明文档

`oct_web_viewer` 是这个仓库里自带的本地网页查看器，用来快速浏览 OCT 体数据、投影图、眼底图、定位线/扫描框以及基础元数据。

它不是独立前端工程，而是一个轻量的 Python 本地服务：

- 后端使用标准库 `http.server`
- 页面是单文件 `HTML + CSS + JavaScript`
- 数据读取和坐标计算复用仓库里的 `oct_converter` 读取器

## 适用场景

- 本地快速查看单个 OCT 文件
- 对比不同厂商格式的体数据和眼底图
- 调试定位线、扫描框、分层轮廓显示
- 验证 `shiwei`、`tupai` 这类目录式 DICOM 数据的解析结果

## 支持的加载模式

查看器支持以下模式：

- `auto`：自动识别
- `heidelberg`：海德堡 `.e2e`
- `topcon`：拓普康 `.fda` / `.fds`
- `zeiss`：蔡司 `.img`
- `optovue`：`.oct`
- `bioptigen`：`.OCT`
- `dicom`：通用 `.dcm` / `.dicom`
- `shiwei`：视微目录式 DICOM
- `tupai`：Tupai 目录式 DICOM

其中有两类模式和普通单文件不一样：

- `shiwei`：需要输入数据目录、该目录中的任一配套 DICOM 文件路径，或可唯一定位到该数据集的上级目录
- `tupai`：需要输入包含 `OCT.dcm` 和 `Fundus.dcm` 的目录、这两个文件中的任意一个路径，或可唯一定位到该数据集的上级目录

这两类模式不支持浏览器中的单文件上传，必须走“路径加载”。

## 启动方式

在仓库根目录执行：

```bash
python web_viewer/oct_web_viewer.py
```

也可以直接指定启动时要加载的文件：

```bash
python web_viewer/oct_web_viewer.py D:\data\sample.e2e
```

如果你更习惯模块方式，也可以：

```bash
python -m web_viewer.oct_web_viewer
```

### 常用参数

```bash
python web_viewer/oct_web_viewer.py ^
  --host 127.0.0.1 ^
  --port 8765 ^
  --img-rows 1024 ^
  --img-cols 512 ^
  --img-interlaced
```

参数说明：

- `path`：可选，启动后自动加载的 OCT 文件路径
- `--host`：服务绑定地址，默认见 `web_viewer/app.py`
- `--port`：服务端口，默认 `8765`
- `--img-rows` / `--img-cols`：用于 Zeiss `.img` 的尺寸
- `--img-interlaced`：按交错格式读取 Zeiss `.img`

启动后，在浏览器打开：

```text
http://<host>:<port>
```

## 页面功能

当前页面主要提供这些能力：

- 加载本地 OCT 数据
- 切换 volume / fundus
- 浏览 B-scan 切片
- 调整对比度、亮度、播放速度
- 查看投影图和眼底定位图
- 显示定位线或扫描框
- 点击眼底图跳转到对应切片
- 查看体数据、眼底图和元数据文本
- 查看分层轮廓（若源文件中包含）
- 记录最近使用路径

当前默认进入后：

- 定位叠加默认开启
- 默认高亮模式是“扫描框”
- “扫描框 / 定位线”按钮顺序为先扫描框、后定位线

## 目录结构

`web_viewer` 目前已按职责拆分，核心文件如下：

- `web_viewer/oct_web_viewer.py`：启动入口包装器
- `web_viewer/app.py`：命令行参数解析和服务启动
- `web_viewer/server.py`：HTTP 路由和请求处理
- `web_viewer/state.py`：数据加载、状态管理、图像编码、overlay 生成
- `web_viewer/shiwei_loader.py`：视微目录式 DICOM 解析
- `web_viewer/tupai_loader.py`：Tupai 目录式 DICOM 解析
- `web_viewer/oct_viewer_vendor_modes.py`：厂商模式、扩展名和错误提示定义
- `web_viewer/oct_web_viewer.html`：页面 UI 与前端逻辑

## 主要数据流

整体流程如下：

1. 页面输入路径，或调用本地选择器
2. 前端请求 `/api/load` 或 `/api/pick-file`
3. `server.py` 把请求转给 `ViewerState`
4. `state.py` 根据 vendor mode 选择：
   - 通用 reader
   - `shiwei_loader`
   - `tupai_loader`
5. 后端生成：
   - volume 数据
   - fundus 数据
   - overlay bounds / segments
   - 可序列化状态 payload
6. 前端根据 payload 渲染 B-scan、眼底图、扫描框和定位线

## HTTP 接口

查看器前后端通过本地 HTTP 接口通信，主要接口如下：

- `GET /`：返回查看器页面
- `GET /api/state`：返回当前状态
- `GET /api/load?path=...&vendor=...`：按路径加载数据
- `GET /api/pick-file?vendor=...`：弹出原生文件选择框并加载
- `POST /api/upload?vendor=...`：浏览器上传文件
- `GET /api/volume/{index}/slice/{slice}.png`：获取某张 B-scan PNG
- `GET /api/volume/{index}/projection.png`：获取投影图
- `GET /api/volume/{index}/contours/{slice}.json`：获取轮廓
- `GET /api/fundus/{index}.png`：获取 fundus 图
- `GET /api/volume/{index}/fundus-view.png`：获取用于叠加显示的眼底视图

## Tupai 模式说明

`tupai` 模式是基于 `scripts/parse_tupai_location.py` 的解析逻辑接入的。

当前行为：

- 接受目录路径，或 `OCT.dcm` / `Fundus.dcm` 路径
- 自动定位同目录下的 `OCT.dcm` 与 `Fundus.dcm`
- 支持输入上级目录，并递归查找唯一匹配的 Tupai 数据集目录
- 读取 Tupai 体数据和 fundus 图
- 读取 frame location，生成定位线
- 从定位线计算扫描范围 `bounds`
- 在前端同时支持：
  - 扫描框模式
  - 定位线模式

相关实现位置：

- `web_viewer/tupai_loader.py`
- `web_viewer/state.py`
- `web_viewer/oct_web_viewer.html`

## 视微模式说明

`shiwei` 模式是目录式 DICOM 加载，不走浏览器单文件上传。

当前行为：

- 输入目录路径即可加载
- 也可以输入目录内任一配套 DICOM 路径
- 也支持输入上级目录，并递归查找唯一匹配的 Shiwei 数据集目录
- 后端会自动回溯到对应目录并补齐所需文件

## 开发和调试建议

如果要继续改这个查看器，建议优先看这些文件：

- 改启动参数：`web_viewer/app.py`
- 改 API：`web_viewer/server.py`
- 改读取/匹配/overlay：`web_viewer/state.py`
- 改目录式 DICOM 解析：`web_viewer/shiwei_loader.py`、`web_viewer/tupai_loader.py`
- 改前端交互和样式：`web_viewer/oct_web_viewer.html`

前端改动后，可用下面方式做一次脚本语法检查：

```bash
node --check extracted.js
```

后端改动后，建议至少执行：

```bash
python -m py_compile web_viewer\__init__.py web_viewer\app.py web_viewer\server.py web_viewer\state.py web_viewer\shiwei_loader.py web_viewer\tupai_loader.py web_viewer\oct_viewer_vendor_modes.py web_viewer\oct_web_viewer.py
```

## 已知约束

- 这是本地调试/查看工具，不是生产部署服务
- 页面和脚本当前仍以单 HTML 文件承载前端逻辑
- 浏览器上传只适用于单文件格式；目录式 DICOM 仍需路径加载
- 某些厂商格式是否有完整定位信息，取决于源文件本身

## 相关文件

- `web_viewer/oct_web_viewer.py`
- `web_viewer/app.py`
- `web_viewer/server.py`
- `web_viewer/state.py`
- `web_viewer/shiwei_loader.py`
- `web_viewer/tupai_loader.py`
- `web_viewer/oct_viewer_vendor_modes.py`
- `web_viewer/oct_web_viewer.html`
