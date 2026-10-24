# LLM Router

LLM Router 是一个本地 OpenAI 兼容路由网关。Codex++ 只需要连接一个本地地址，Router 再按模型名把请求转发到不同上游，并在 OpenAI Responses API 与 Chat Completions API 之间转换协议。

默认监听地址：`http://127.0.0.1:8765/v1`

## 功能

- 按模型名选择供应商：模型名统一写成 `<供应商>/<模型名>`，同名模型不会互相混淆。
- 聚合上游模型列表，支持手动补充、隐藏模型和显式模型映射。
- 同时接收 `/v1/responses`、`/v1/chat/completions`、`/v1/images/generations` 和 `/v1/images/edits`。
- 按 `model_map` 或供应商模型列表，把生图请求转发到配置的图片供应商，技能侧只需连接本地 Router。
- 在 Responses 与 Chat 协议之间转换文本、推理内容、用量统计、图片输入和 JSON Schema 输出格式。
- 支持为纯文本模型配置图片省略（`strip`）或视觉模型描述辅助（`vlm`）。
- 桥接 function、custom、namespace 和 tool search 工具调用。
- 针对 MiniMax M3 的 Responses 子集提供协议兼容：把 Codex 的 `tool_search` / `custom` / `agent_message` 历史降级为标准 `function` 调用，并只在自己返回内容安全 1026 时单次降级图片。
- 支持按供应商透传 Codex `prompt_cache_key`，用于能识别该字段的第三方 Responses 上游，避免因 Router 统一剥离导致缓存命中率过低。
- 在响应开始前对网络错误、429 和临时上游 5xx 做有限重试。
- 内置 Agent 接入预设：Codex 桌面版（Responses）、DeepSeek Harness（Chat）、Claude Code（Anthropic Messages）可同时接入，互不影响。
- 新增 Anthropic Messages 入站（`/v1/messages`），自动与 Chat / Responses 上游互转（system、tool_use / tool_result、thinking、图片块）。
- 支持为每个 Agent 设置默认模型（仅客户端未指定模型时使用）与推理档位（仅请求未携带档位时生效），控制台一键复制接入配置并测试连接；客户端选择的模型与档位一律原样生效，不强制改写。
- 对非原生 OpenAI Responses 上游默认把 Codex 的 `agent_message` 转成普通 user 消息，把 `encrypted_content` 中的任务文本解包为 `input_text`，并移除 `author` / `recipient`、`prompt_cache_key` 等会被第三方网关拒绝的 Codex 字段；无法解包的私有字段会降级为占位文本。
- 提供本地管理界面，用于维护供应商、模型路由、价格、运行统计和 Agent 接入。
- 概览页按时间范围统计请求数、Tokens、输出速度（输出 token ÷ 生成耗时，已扣除首字）、平均首字、平均耗时和缓存命中率（默认最近 12 小时，可选「全部」）；「最近请求」可以只看正常或错误请求，并显示首字耗时。

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
   ├─ suppliers/             # 供应商专项适配
   │  ├─ shared.py           # 共享工具函数
   │  ├─ anthropic_messages.py # Anthropic Messages 协议适配（Claude Code）
   │  └─ minimax.py          # MiniMax Responses 兼容
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
- `responses_compat`：Responses 上游的专用兼容模式，可选 `""` 或 `"minimax"`；只有 MiniMax M3 供应商需要设置为 `"minimax"`。
- `forward_prompt_cache_key`：是否把 Codex 的 `prompt_cache_key` 原样透传给上游；仅在上游确实支持缓存键时开启，默认关闭。
- `model_map`：公开模型名到供应商名的显式映射，键就是 `<供应商>/<模型名>`（值只作标注，选路只看键）。
- `default_supplier`：已废弃。Router 不再按默认供应商兜底，模型名必须带供应商前缀。
- `model_blacklist`：不在合并模型列表中展示的模型。
- `image_handling`：按模型配置图片处理模式，取值为 `send-as-is`、`strip` 或 `vlm`。
- `agent_profiles`：Agent 接入预设。`default_model` 为该 Agent 的默认模型，仅当客户端未指定模型时填入（客户端选的模型永远原样使用）；`reasoning_effort` 为该 Agent 的推理档位，仅当请求未显式携带档位时生效。推理强度不再按模型（`model_reasoning_efforts`）固定注入；按模型的兼容覆盖（`model_overrides`）已移除，`openai_sanitize` / `forward_prompt_cache_key` 请在供应商上设置。

### 模型命名：`<供应商>/<模型名>`

多个渠道可能提供同名模型（例如 `deepseek-v4.1-flash`），所以 Router 对外暴露的模型名统一带上供应商：

- `/v1/models` 返回的 `id` 是 `<供应商>/<模型名>`，`owned_by` 是对应供应商；
- 模型名本身带 `/` 时原样保留，只在最前面加供应商：`xkiro` 渠道的 `z-ai/glm-5.3` 显示为 `xkiro/z-ai/glm-5.3`；
- Agent 里直接填这个公开名即可。Router 只在**第一个** `/` 处切分，剩下的部分作为上游模型名转发，上游永远收不到供应商前缀；
- 选路只看名字：只在第一个 `/` 处切分，前半段必须是已配置的供应商，后半段原样转发给上游；没有别名表，也没有默认供应商兜底；
- 配置在启动时自动迁移：`model_map`、`model_prices`、`model_max_output_tokens`、`agent_profiles.*.default_model` 里的裸模型名会被补成 `<供应商>/<模型名>`（只迁移一次，写回 `config.json`）。

同名模型可以同时挂在多个供应商下：公开名带供应商前缀，所以 `pixel/deepseek-v4.1-flash` 与 `xkiro/deepseek-v4.1-flash` 互不冲突；在模型路由页选择供应商后重复添加即可。

## Agent 接入（一键切换）

Router 的三类协议端点常开，Codex 桌面版、DeepSeek Harness 和 Claude Code 可以同时接入，互不影响：

| Agent | 端点 | 协议 |
|---|---|---|
| Codex 桌面版 / Codex++ | `/v1/responses` | OpenAI Responses |
| DeepSeek Harness | `/v1/chat/completions` | OpenAI Chat Completions |
| Claude Code | `/v1/messages` | Anthropic Messages |

控制台「Agent 接入」页提供：

- 每个 Agent 的最近请求状态与 24h 用量（按端点和 User-Agent 自动识别归因）；
- 「默认模型」：仅当客户端未指定模型时使用，客户端选择的模型原样生效、不改写；
- 「推理档位」覆盖：请求未显式携带档位时生效；
- 一键复制接入配置（Codex Switch 供应商参数 / DeepSeek Harness 的 `cordis.patch.yml` 片段 / Claude Code 的环境变量）；
- 「测试连接」：从 Router 本机按该 Agent 的协议发一条最小请求，端到端验证。

### Claude Code 接入示例

```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8765"
$env:ANTHROPIC_AUTH_TOKEN = "local-router"   # 任意非空即可
claude
```

`POST /v1/messages/count_tokens` 提供本地粗估的输入 token 数，不请求上游。

## 接入 Codex++

在 Codex++ 中新增一个 OpenAI Responses 兼容供应商：

- 接口地址：`http://127.0.0.1:8765/v1`
- API Key：任意非空占位值
- 协议：Responses

切换模型时，Router 只在第一个 `/` 处切分模型名：前半段必须是已配置的供应商，后半段原样转发给上游；模型名请使用 `/v1/models` 返回的 `<供应商>/<模型名>`。

### Codex Switch 模型元数据

要让 Codex Desktop 的 UI 显示原生思维摘要和文件变更卡片，Codex Switch 侧的第三方模型元数据需要与协议能力一致：

- `apply_patch_tool_type` 设为 `"freeform"`。否则 Codex 不会声明 `apply_patch` freeform 工具，第三方模型只能把补丁包进 `exec_command`，UI 会显示为命令执行而不是 `FileChange`。
- `default_reasoning_summary` 设为 `"auto"`。`none` 会明确要求上游不返回摘要；`supports_reasoning_summaries` 仍需保持 `true`。

Router 会把 Codex 的 `custom_tool_call`（如 `apply_patch`）桥接为 Chat function call，并在响应流中恢复成 Responses `custom_tool_call`；但如果客户端没有声明该工具，Router 不能安全地替客户端伪造工具调用。

生图技能也走同一个入口：在 `suppliers` 中添加图片供应商，并把 `gpt-image-2` 映射到该供应商。技能调用 `http://127.0.0.1:8765/v1/images/generations` 或 `/v1/images/edits`，Router 读取供应商配置并转发，技能不需要知道上游地址或密钥。

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

- `GET /v1/models`（`id` 为 `<供应商>/<模型名>`）
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/messages`（Anthropic Messages，Claude Code 接入）
- `POST /v1/messages/count_tokens`
- `POST /v1/images/generations`
- `POST /v1/images/edits`

管理 API：

- `GET /`：本地管理界面
- `GET /api/health`
- `GET/PUT /api/config`
- `POST/PUT/DELETE /api/suppliers`
- `GET/PUT /api/model_map`
- `POST /api/test-supplier`
- `GET /api/merged-models`
- `POST /api/fetch-models`
- `GET /api/agents`：Agent 接入预设与状态
- `PUT /api/agents/{agent}`：保存单个 Agent 的默认模型 / 推理档位
- `POST /api/agents/{agent}/test`：按该 Agent 的协议端到端测试
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
