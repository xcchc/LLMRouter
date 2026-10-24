#!/usr/bin/env bash
# 在 macOS 上本地打包 LLMRouter.app
#
# 用法：
#   ./build-macos.sh          # 构建
#   ./build-macos.sh --zip    # 构建并打 zip（发给别人用）
#
# 产物：dist/LLMRouter.app，加 --zip 时还会生成 LLMRouter-macos-<arch>.zip
set -euo pipefail

cd "$(dirname "$0")"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "错误：这个脚本只能在 macOS 上运行（PyInstaller 不能交叉编译 .app）。" >&2
  echo "在 Windows 上请用 build.bat，macOS 版由 GitHub Actions 构建。" >&2
  exit 1
fi

PYTHON="${PYTHON:-python3}"

echo "==> 检查 Python"
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), sys.version'

if [[ ! -d .venv ]]; then
  echo "==> 创建虚拟环境 .venv"
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> 安装依赖"
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements-build.txt

echo "==> 跑单元测试"
python -m unittest discover -s tests

echo "==> PyInstaller 打包"
python -m PyInstaller --noconfirm --clean LLMRouter.spec

if [[ ! -d dist/LLMRouter.app ]]; then
  echo "错误：没有生成 dist/LLMRouter.app" >&2
  exit 1
fi

echo "==> 自检：打包后的关键依赖能不能 import"
LLMROUTER_SELFTEST=1 ./dist/LLMRouter.app/Contents/MacOS/LLMRouter

echo "==> 冒烟测试（无界面模式启动网关，检查 /api/health）"
ARCH="$(uname -m)"
[[ "$ARCH" == "arm64" ]] && ARCH_LABEL="arm64" || ARCH_LABEL="x64"
SMOKE_PORT=8799
LLMROUTER_HEADLESS=1 ./dist/LLMRouter.app/Contents/MacOS/LLMRouter &
SMOKE_PID=$!
cleanup() { kill "$SMOKE_PID" 2>/dev/null || true; }
trap cleanup EXIT

ok=0
for _ in $(seq 1 40); do
  if curl -fsS "http://127.0.0.1:${SMOKE_PORT}/api/health" >/dev/null 2>&1 \
     || curl -fsS "http://127.0.0.1:8765/api/health" >/dev/null 2>&1; then
    ok=1
    break
  fi
  sleep 0.5
done
if [[ "$ok" != "1" ]]; then
  echo "错误：打包后的程序没能启动网关" >&2
  exit 1
fi
echo "    网关启动正常"
cleanup
trap - EXIT

if [[ "${1:-}" == "--zip" ]]; then
  ZIP_NAME="LLMRouter-macos-${ARCH_LABEL}.zip"
  echo "==> 打包 $ZIP_NAME"
  rm -f "$ZIP_NAME"
  # ditto 才能保住符号链接和可执行权限，zip 命令会破坏 .app
  ditto -c -k --sequesterRsrc --keepParent dist/LLMRouter.app "$ZIP_NAME"
  echo "    完成：$ZIP_NAME（$(du -h "$ZIP_NAME" | cut -f1)）"
fi

echo
echo "构建完成：dist/LLMRouter.app"
echo "首次打开：右键点图标 →「打开」，或执行"
echo "  xattr -dr com.apple.quarantine /Applications/LLMRouter.app"
