# OCT / FA Unified Web Service

这个服务是给其他网页、前端应用或局域网内系统调用的统一后端，不只是内置 viewer 使用。

## 启动

在仓库根目录执行：

```bash
python web_viewer/web_service.py --host 127.0.0.1 --port 8765
```

如果已经按包安装，也可以直接执行：

```bash
oct-web-service --host 127.0.0.1 --port 8765
```

启动后可访问：

- API 文档：`http://127.0.0.1:8765/docs`
- 健康检查：`http://127.0.0.1:8765/healthz`
- API 根：`http://127.0.0.1:8765/api`
- OCT viewer：`http://127.0.0.1:8765/oct`
- FA viewer：`http://127.0.0.1:8765/fa`

## 跨域

默认允许所有来源跨域访问。

如果只想允许指定网页来源：

```bash
python web_viewer/web_service.py ^
  --cors-allow-origin http://localhost:3000 ^
  --cors-allow-origin http://127.0.0.1:5173
```

## OCT API

### 1. 查看当前状态

```http
GET /api/oct/state
```

返回当前是否已加载、数据源路径、volume 列表、fundus 列表等。

### 2. 从本地路径加载

```http
GET /api/oct/load?path=D:/data/test.e2e&vendor=auto
```

支持 vendor：

- `auto`
- `heidelberg`
- `topcon`
- `shiwei`
- `tupai`

### 3. 上传单个文件

```http
POST /api/oct/upload?vendor=auto
Content-Type: multipart/form-data
```

表单字段名必须是 `file`。

说明：

- 这个接口依赖 `python-multipart`
- 如果环境没装该依赖，会返回 `503`

### 4. 获取某个切片 PNG

```http
GET /api/oct/volume/{volume_index}/slice/{slice_index}.png?contrast=100&brightness=0
```

### 5. 获取 projection PNG

```http
GET /api/oct/volume/{volume_index}/projection.png
```

### 6. 获取 contours JSON

```http
GET /api/oct/volume/{volume_index}/contours/{slice_index}.json
```

### 7. 获取 fundus PNG

```http
GET /api/oct/fundus/{fundus_index}.png
```

### 8. 获取 fundus overlay PNG

```http
GET /api/oct/volume/{volume_index}/fundus-view.png
```

## FA API

### 1. 查看当前状态

```http
GET /api/fa/state
```

### 2. 从本地路径加载

```http
GET /api/fa/load?path=D:/data/fa_case&vendor=auto
```

支持 vendor：

- `auto`
- `topcon`
- `zeiss`
- `hdb`

### 3. 通过系统文件选择器加载

```http
GET /api/fa/pick-file?vendor=auto
```

### 4. 通过系统目录选择器加载

```http
GET /api/fa/pick-directory?vendor=auto
```

### 5. 获取帧 PNG

```http
GET /api/fa/frame/{frame_index}.png?contrast=100&brightness=0
```

## 错误返回

普通错误一般返回：

```json
{
  "error": "..."
}
```

常见状态码：

- `200` 成功
- `400` 参数错误或读取失败
- `404` 路由不存在
- `503` 可选依赖缺失，例如上传接口缺少 `python-multipart`

## 前端调用示例

### 加载 OCT 文件

```js
const baseUrl = "http://127.0.0.1:8765";

async function loadOct(path) {
  const url = `${baseUrl}/api/oct/load?path=${encodeURIComponent(path)}&vendor=auto`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}
```

### 读取 OCT 状态

```js
async function getOctState() {
  const response = await fetch("http://127.0.0.1:8765/api/oct/state");
  return response.json();
}
```

### 在页面中显示第一个切片

```html
<img src="http://127.0.0.1:8765/api/oct/volume/0/slice/0.png" alt="OCT slice">
```

### 在页面中显示第一张 FA 帧图

```html
<img src="http://127.0.0.1:8765/api/fa/frame/0.png" alt="FA frame">
```

## 示例页面

仓库里提供了一个最小接入示例：

- `examples/web_service_client_example.html`

它演示了：

- 通过输入路径调用 `/api/oct/load`
- 读取 `/api/oct/state`
- 在普通网页里直接显示 OCT 切片图

