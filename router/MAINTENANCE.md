# LLM Router 维护说明

## 运行原则

- Router 只由用户手动启动，不注册开机自启动，也不依赖外部守护进程自动拉起。
- 日常运行可以使用已发布的 `LLMRouter.exe`，也可以双击 `start.bat` 从源码运行。
- 桌面窗口关闭后可能隐藏到系统托盘；需要停止服务时，从托盘菜单选择“退出软件”。
- 修改监听端口或替换运行版前，先完全退出当前进程。完成操作后仍由用户手动启动。

## 本地文件

- `config.json`：真实运行配置，包含 API Key，禁止提交或外传。
- `stats.jsonl`：本地运行统计，缺失时程序会自动创建。
- `crash.log` 和其他日志：仅用于本地排错，不进入版本控制或发布包。
- `LLMRouter.exe`：本机手动运行版，作为构建产物忽略；公开二进制应通过 Release 单独发布。
- `LLMRouter.new.exe` / `LLMRouter.previous.exe`：一键更新时使用的暂存与备份文件，更新完成后会被清理。
- `.venv/`、`build/`、`dist/`、`__pycache__/`：可重新生成，不进入版本控制。

## 开发文件

- `app.py`：桌面窗口、托盘和服务生命周期。
- `router.py`：网关、协议转换和管理 API。
- `stats.py`：统计模块。
- `vision.py`：按模型处理图片（`send-as-is` / `strip` / `vlm`）。
- `dashboard.html`、`icon.ico`：界面和打包资源。
- `config.example.json`：不含真实密钥的公开配置模板。
- `tests/`：协议转换和流式行为测试。
- `LLMRouter.spec`：PyInstaller 打包配置。
- `start.bat`：源码版手动启动入口。
- `build.bat`：测试与打包入口，产物位于 `dist\LLMRouter.exe`。

## 测试与打包

运行测试：

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests
```

构建发布版：

```powershell
.\build.bat
```

`build.bat` 会运行测试并生成 `dist\LLMRouter.exe`，但不会替换根目录运行版，也不会自动启动 Router。维护者应在当前进程完全退出后手动验证新产物。

需要在 Router 正在使用时准备更新，可把新文件暂存为 `LLMRouter.new.exe`，然后在 EXE 版设置页点击“更新并重启”：脚本会等待旧进程退出、替换文件、校验哈希并启动新版。源码模式仍使用 `apply-update.bat` 手动兜底。

## 一键更新流程（EXE 版）

- 将构建产物复制为同目录 `LLMRouter.new.exe`。
- 设置页点击“检查更新”，确认发现新版后点击“更新并重启”。当前进程退出后，脚本把旧版备份为 `LLMRouter.previous.exe`，复制新版并校验 SHA-256，清理 `crash.log` 后启动新版。
- 任一环节失败时脚本会恢复旧版并写入 `update-error.log`；新版校验失败时不会覆盖旧版。
- 源码模式（`start.bat`）不启用一键更新，请继续使用 `apply-update.bat`。

## 发布前检查

1. 选择开源许可证并在仓库根目录添加 `LICENSE`。
2. 运行完整测试，确认协议转换、工具续轮和流式事件通过。
3. 搜索 API Key、Authorization 头、个人绝对路径、私有域名及供应商名称。
4. 确认 `config.json`、`stats.jsonl`、日志、数据库、备份和临时目录均未进入提交。
5. 清理 `.venv/`、`build/`、`dist/`、缓存和本地打包产物。
6. 从干净目录重新构建，并使用不含真实密钥的临时配置做启动验证。
7. 单独打包 Release，确认发布包不包含本机配置和运行统计。

## 协议兼容边界

- Chat 上游只有在正确实现工具调用、流式增量和推理字段时，才能接近原生 Responses 体验。
- Codex MultiAgent V2 会向第三方 Responses 上游发送 OpenAI 私有的 `agent_message.encrypted_content`。Router 不能解密它；非原生 OpenAI 上游应保持 `openai_sanitize` 为缺省 `true`，让 Router 在请求转发前降级为普通 user 文本。
- `web_search` 等服务端托管工具需要独立后端，Router 不会伪造执行结果。
- 未知密钥生成的 `agent_message.encrypted_content` 无法由 Router 解密。
- 图片、音频和内嵌文件是否可用取决于上游能力；图片可通过 `image_handling` 配置为 `send-as-is`、`strip` 或 `vlm`，纯文本模型可用省略或视觉辅助描述继续处理图片。
- Chat 没有原生 `phase`，首段文本需要缓冲后再判定 commentary 或 final answer。
- 流式响应开始后的上游中断不能透明重试，以免重复输出或重复执行工具。

## 本地路由配置

不要在维护文档中记录个人供应商、私有域名、真实模型路由或 API Key。示例应使用 `responses-provider`、`chat-provider` 和 `*-example` 等占位名称；真实映射只保存在被忽略的 `config.json` 中。
