# LLM Router

LLM Router 是一个本地 OpenAI 兼容路由网关。Codex++ 只需要连接一个本地地址，Router 再按模型名把请求转发到不同上游，并在 OpenAI Responses API 与 Chat Completions API 之间转换协议。

默认监听地址：`http://127.0.0.1:8765/v1`

## 主要能力

- 按模型名选择供应商，也可配置默认供应商兜底。
- 聚合上游模型列表，并支持手动补充或隐藏模型。
- 同时接收 `/v1/responses` 和 `/v1/chat/completions` 请求。
- 在 Responses 与 Chat 协议之间转换文本、推理内容、用量统计、图片输入和 JSON Schema 输出格式。
- 支持为纯文本模型配置图片省略（strip）或视觉模型描述辅助（VLM）。
- 桥接 function、custom、namespace 和 tool search 工具调用，并尽量还原 Codex 的过程折叠、文件修改卡和 diff 展示。
- 在响应开始前对网络错误、429 和临时上游 5xx 做有限重试。
- 提供本地管理界面，用于维护供应商、模型路由、价格与运行统计。

## 兼容性说明

Router 的目标是让 DeepSeek 和其他 Chat 协议模型在 Codex 中尽可能接近原生 GPT 模型的体验，但最终效果仍取决于上游模型和供应商是否正确支持工具调用、流式输出、推理字段及结构化输出。

已经通过真实 DeepSeek Chat 上游验证的路径包括：reasoning、commentary/final phase、function、custom `apply_patch`、namespace、延迟 `tool_search`、工具续轮、文件修改卡、JSON Schema 回退和临时 5xx / 429 重试。

以下能力不能仅靠协议转换完整复刻：

- Codex MultiAgent V2 会把子代理任务放进 `agent_message.encrypted_content`。真实 DeepSeek 抓包中该字段就是任务明文，因此 Router 默认把 `agent_message` 转成普通 user 消息，并把 `encrypted_content` 解包为 `input_text`；只有遇到确实无法解包的 OpenAI 私有密文时才替换为占位说明。非原生 OpenAI Responses 上游应保持 `openai_sanitize` 为缺省 `true`。
- `web_search` 等由 OpenAI 服务端执行的托管工具，需要额外接入真实的搜索或工具后端。Router 只会向上游说明能力限制，不会伪造成功结果。
- 某些 Chat 上游不接受强制 `tool_choice`，或会返回非标准工具调用字段。这类差异需要由具体供应商适配。
- 图片会映射为 Chat `image_url`。Router 支持按模型配置三种处理模式：`send-as-is` 原样发送、`strip` 替换为文本占位、`vlm` 先交给视觉辅助供应商生成描述再注入请求。仅接受文本的模型应使用 `strip` 或 `vlm`。
- Responses 的内嵌文件和音频没有通用的 Chat 等价格式，目前不能保证保真转换。Codex 读取本地文件时应优先使用客户端文件工具。
- Chat 流没有原生 `phase`。Router 需要等到文本结束或出现工具调用后，才能可靠区分 commentary 与 final answer，因此首段文本的显示时机无法做到与原生 Responses 完全一致。
- 上游在流式响应开始后中断时，Router 不能透明重试整次生成，否则可能产生重复文本或重复工具调用。

## 手动启动

Router 不注册 Windows 开机自启动，进程退出后也不会自动拉起。需要使用时由用户手动启动。

### 从源码运行

当前维护和测试环境使用 Windows 与 Python 3.12。

1. 双击 `start.bat`。
2. 首次运行会创建 `.venv` 并安装 `requirements.txt`。
3. 浏览器模式下访问 `http://127.0.0.1:8765/`；桌面模式会直接打开管理窗口。

桌面窗口关闭后通常会隐藏到系统托盘。要完全停止 Router，请在托盘菜单中选择“退出软件”。

### 运行发布版

如果从 Releases 下载了 `LLMRouter.exe`，双击该文件即可手动启动。运行数据会保存在 EXE 同目录。仓库本身不依赖 `dist\LLMRouter.exe`，构建产物不应提交到源码仓库。

## 首次配置

首次启动会自动创建空的 `config.json`。也可以在启动前复制公开模板：

```powershell
Copy-Item config.example.json config.json
```

然后通过管理界面或本地 `config.json` 配置供应商。`config.json` 包含 API Key，已加入 `.gitignore`，不得提交或公开。

常用字段：

- `port`：本地监听端口，默认 `8765`。
- `open_browser`：是否使用浏览器模式。
- `suppliers`：供应商列表，包含名称、接口地址、API Key、上游协议和可选模型列表。
- `wire_api`：上游协议，可选 `responses` 或 `chat`。
- `openai_sanitize`：是否在转发到非原生 OpenAI Responses 上游前清理 Codex 的 `agent_message` / `encrypted_content`；缺省为 `true`，仅对 Responses 上游生效。
- `image_handling`：供应商内按模型名配置图片处理模式，取值为 `send-as-is`、`strip` 或 `vlm`；缺省为 `send-as-is`。
- `vlm_supplier`：`vlm` 模式使用的视觉辅助供应商名称，复用其 `base_url` 和 `api_key`。
- `vlm_model`：视觉辅助供应商使用的视觉模型名，例如 `gpt-5.6-luna`。
- `model_map`：模型名到供应商名的显式映射。
- `default_supplier`：没有显式映射时使用的默认供应商。
- `model_blacklist`：不在合并模型列表中展示的模型。

除监听端口外，配置会在请求时重新读取。修改端口后，请完全退出 Router，再手动启动。

## 接入 Codex++

在 Codex++ 中新增一个 OpenAI Responses 兼容供应商：

- 接口地址：`http://127.0.0.1:8765/v1`
- API Key：任意非空占位值
- 协议：Responses

Codex++ 始终连接这个本地供应商。切换模型时，Router 根据 `model_map`、供应商模型列表和 `default_supplier` 选择上游。

## 图片处理模式

非多模态上游（例如 DeepSeek）收到图片时可能直接报错。在供应商编辑弹窗中，每个模型都有一行“图片处理模式”下拉选择，可选：

- `send-as-is`：原样发送图片（默认），适合真正支持多模态的上游。
- `strip`：把图片块替换为 `[图片已省略]`，纯文本模型仍可继续回答。
- `vlm`：把图片分批发送给视觉辅助供应商，生成描述后替换为 `[图片描述] ...`，再交给原上游。

选择 `vlm` 后还需在下拉框中选择“视觉辅助供应商”，并从该供应商已获取的模型中选择“视觉模型”。视觉辅助供应商复用现有供应商的 `base_url` 和 `api_key`，不需要单独保存 VLM Key；描述结果有 24 小时内存缓存。视觉服务失败时请求会以明确占位文本继续，不会整单失败。

## 开发与测试

安装运行依赖后执行测试：

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests
```

重新打包：

```powershell
.\build.bat
```

`build.bat` 会安装构建依赖、运行测试，并把临时产物写入 `build\` 和 `dist\`。发布二进制前应先退出正在运行的 Router；构建完成后由维护者手动验证和发布，脚本不会替换或启动根目录的运行版。

若根目录已有正在运行的旧版，可先把新产物命名为 `LLMRouter.new.exe`，再到设置页“更新”卡片点击“检查更新”和“更新并重启”。EXE 版会自动替换旧 EXE 并启动新版；源码模式仍使用 `apply-update.bat` 作为手动兜底。

从 2026.08.10 起，`build.bat` 会在测试和打包后自动生成 `LLMRouter.new.exe`，可直接用于设置页内置更新：

1. 构建完成后不要关闭运行中的旧版 Router。
2. 打开管理界面，进入“设置”->“更新”->“检查更新”。
3. 页面显示已发现 `LLMRouter.new.exe` 后点击“更新并重启”。


## 目录说明

- `app.py`：桌面窗口、系统托盘与本地服务启动器。
- `router.py`：路由、协议转换和管理 API。
- `stats.py`：本地统计模块。
- `vision.py`：按模型处理图片（`send-as-is` / `strip` / `vlm`）。
- `dashboard.html`：管理界面。
- `config.example.json`：可公开的配置模板。
- `config.json`：本地真实配置，含 API Key，不提交。
- `stats.jsonl`：本地运行统计，不提交。
- `tests/`：协议转换与流式事件测试。
- `LLMRouter.spec`：PyInstaller 构建配置。
- `start.bat`：源码版手动启动入口。
- `build.bat`：测试和打包入口。
- `apply-update.bat`：手动替换 EXE 的兜底脚本；EXE 版可在设置页使用一键“更新并重启”。

## 开源前安全检查

提交或发布前至少确认：

- `config.json`、`.env`、日志、统计文件、数据库和备份目录没有进入版本控制。
- API Key、Authorization 头、个人绝对路径和私有供应商名称没有出现在源码、文档、测试输出或打包资源中。
- `.venv/`、`build/`、`dist/`、缓存和本地 EXE 没有进入源码提交。
- 发布包中只包含运行所需文件，不附带本机 `config.json`、`stats.jsonl` 或 `crash.log`。

## 常见问题

**模型没有可用上游**

检查模型是否存在于供应商模型列表或 `model_map` 中，并确认已经配置 `default_supplier`。

**某个模型报错，其他模型正常**

检查该供应商的 `wire_api`、接口地址、API Key，以及上游是否支持当前请求中的工具或结构化输出。

**模型列表为空**

部分上游不允许查询 `/v1/models`。可在供应商配置中手动填写模型名。

**窗口关闭后端口仍在监听**

桌面窗口关闭后可能只是隐藏到托盘。请从托盘菜单完全退出。

**启动后立即退出**

检查同目录 `crash.log`。源码版还可以在终端运行 `start.bat` 查看错误信息。
