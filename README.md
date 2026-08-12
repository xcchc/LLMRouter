# LLM Router

LLM Router 是一个本地 OpenAI 兼容路由网关。Codex++ 只需要连接一个本地地址，Router 再按模型名把请求转发到不同上游，并在 OpenAI Responses API 与 Chat Completions API 之间转换协议。

默认监听地址：`http://127.0.0.1:8765/v1`

## 功能

- 按模型名选择供应商，也可配置默认供应商兜底。
- 聚合上游模型列表，支持手动补充、隐藏模型和显式模型映射。
- 同时接收 `/v1/responses` 和 `/v1/chat/completions`。
- 在 Responses 与 Chat 协议之间转换文本、推理内容、用量统计、图片输入和 JSON Schema 输出格式。
- 支持为纯文本模型配置图片省略（`strip`）或视觉模型描述辅助（`vlm`）。
- 桥接 function、custom、namespace 和 tool search 工具调用。
- 在响应开始前对网络错误、429 和临时上游 5xx 做有限重试。
- 对非原生 OpenAI Responses 上游默认把 Codex 的 `agent_message` 转成普通 user 消息，把 `encrypted_content` 中的任务文本解包为 `input_text`，并移除 `author` / `recipient`、`prompt_cache_key` 等会被第三方网关拒绝的 Codex 字段；无法解包的私有字段会降级为占位文本。
- 提供本地管理界面，用于维护供应商、模型路由、价格和运行统计。

## 目录结构

```text
codex-api/
├─ README.md
├─ LICENSE
├─ .gitignore
└─ router/
   ├─ app.py                 # 桌面窗口、系统托盘与本地服务启动器
   ├─ router.py              # 路由、协议转换和管理 API
   ├─ vision.py              # 图片处理策略
   ├─ stats.py               # 本地运行统计
   ├─ dashboard.html         # 本地管理界面
   ├─ config.example.json    # 公开配置模板
   ├─ tests/                 # 单元测试
   ├─ start.bat              # 源码版启动入口
   ├─ build.bat              # 测试和打包入口
   └─ README.md              # 详细维护和配置说明
```

## 快速开始

### 从源码运行

当前维护和测试环境使用 Windows 与 Python 3.12。

```powershell
cd router
.\start.bat
```

首次运行会创建 `.venv` 并安装 `requirements.txt`。浏览器模式下访问：

```text
http://127.0.0.1:8765/
```

### 运行发布版

下载或构建 `LLMRouter.exe` 后直接双击即可。运行数据会保存在 EXE 同目录。

## 配置

首次启动会自动创建空的 `config.json`，也可以复制公开模板：

```powershell
Copy-Item config.example.json config.json
```

常用配置字段：

- `port`：本地监听端口，默认 `8765`。
- `open_browser`：是否使用浏览器模式。
- `suppliers`：上游供应商列表，包含名称、接口地址、API Key、协议和可选模型列表。
- `wire_api`：上游协议，可选 `responses` 或 `chat`。
- `openai_sanitize`：是否在转发到非原生 OpenAI Responses 上游前转换 Codex 的 `agent_message` 并解包 `encrypted_content`；缺省为 `true`。
- `model_map`：模型名到供应商名的显式映射。
- `default_supplier`：没有显式映射时使用的默认供应商。
- `model_blacklist`：不在合并模型列表中展示的模型。
- `image_handling`：按模型配置图片处理模式，取值为 `send-as-is`、`strip` 或 `vlm`。

## 接入 Codex++

在 Codex++ 中新增一个 OpenAI Responses 兼容供应商：

- 接口地址：`http://127.0.0.1:8765/v1`
- API Key：任意非空占位值
- 协议：Responses

切换模型时，Router 根据 `model_map`、供应商模型列表和 `default_supplier` 选择上游。

## 更新

### 构建更新包

```powershell
git clone https://github.com/xcchc/LLMRouter.git
cd LLMRouter\router
.\build.bat
```

构建完成后会生成两个文件：

- `router\dist\LLMRouter.exe`：可用于发布的新版完整程序。
- `router\LLMRouter.new.exe`：设置页“更新并重启”可直接识别的更新包。

`build.bat` 构建完成后会自动生成 `router\LLMRouter.new.exe`，不需要手动改名或提交 Git；把它留在 Router 同目录，设置页就能直接检测到并更新。`LLMRouter.new.exe` 不会进入 Git 仓库，避免直接替换正在运行的旧版。

### 在设置页更新

1. 打开运行中 Router 的管理界面，例如 `http://127.0.0.1:8765/`。
2. 进入“设置”，打开“更新”卡片。
3. 点击“检查更新”，确认提示已经找到 `LLMRouter.new.exe`。
4. 点击“更新并重启”，程序会自动替换 `LLMRouter.exe` 并启动新版。

如果运行的是源码模式而不是 EXE，请使用 `router\apply-update.bat` 手动兜底替换。

### 发布给其他用户

需要分发时，把 `router\dist\LLMRouter.exe` 上传到 GitHub Releases。使用者下载后，将其放到正在运行的 Router 同目录并命名为 `LLMRouter.new.exe`，然后按上面的设置页步骤更新。

## API

对外 API：

- `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions`

管理 API：

- `GET /`：本地管理界面
- `GET /api/health`
- `GET/PUT /api/config`
- `POST/PUT/DELETE /api/suppliers`
- `GET/PUT /api/model_map`
- `POST /api/test-supplier`
- `GET /api/merged-models`
- `POST /api/fetch-models`
- `GET /api/stats/summary`
- `GET /api/stats/raw`
- `PUT/DELETE /api/model-price`
- `POST /api/reload`
- `GET /api/update/status`
- `POST /api/update/apply`

## 协议转换

Router 的核心工作是让 Responses 客户端可以连接 Chat 上游，也让 Chat 客户端可以连接 Responses 上游。

### Responses -> Chat

- `instructions` 转换为 system 消息。
- `reasoning` 转换为 Chat 的 `reasoning_content`。
- `function_call`、`custom_tool_call`、`tool_search_call` 转换为 `tool_calls`。
- `function_call_output` 转换为 `tool` 消息。
- `max_output_tokens` 转换为 `max_completion_tokens`。
- namespace 工具转换为 `namespace__name` 扁平名称。
- JSON Schema 输出转换为 Chat 的 `response_format`。

### Chat -> Responses

- system/developer 消息合并为 `instructions`。
- `reasoning_content` 转换为独立的 `reasoning` 输出项。
- `tool_calls` 转换为 `function_call`。
- `tool` 消息转换为 `function_call_output`。
- `response_format` 转换为 `text.format`。
- 流式 Chat 响应转换为 Responses SSE 事件。

## 图片处理

非多模态上游收到图片时可能直接报错，可以在供应商编辑弹窗中为每个模型配置图片处理模式：

- `send-as-is`：原样发送图片，适合真正支持多模态的上游。
- `strip`：把图片块替换为文本占位，纯文本模型仍可继续回答。
- `vlm`：把图片交给视觉辅助供应商生成描述，再注入原请求。

## 开发与测试

安装运行依赖后执行测试：

```powershell
cd router
.venv\Scripts\python.exe -m unittest discover -s tests
```

重新打包：

```powershell
cd router
.\build.bat
```

## 安全

`config.json` 包含真实 API Key，已加入 `.gitignore`，不要提交或公开。管理 API 没有鉴权，服务只监听 `127.0.0.1`，不要暴露到局域网或公网。

## License

MIT License。完整文本见 [LICENSE](LICENSE)。
