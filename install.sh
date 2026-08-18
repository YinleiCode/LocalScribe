#!/usr/bin/env bash
# =============================================================================
# LocalScribe · 一键安装脚本(macOS Apple Silicon)
# =============================================================================
# 流程:
#   1. 检查 / 安装系统依赖(ffmpeg, uv, pnpm, Rust)
#   2. 创建 Python venv 并装 sidecar 依赖
#   3. 下载 Whisper 模型(1.5 GB,首次需要)
#   4. 装前端依赖
#   5. 构建 .app 到 src-tauri/target/release/bundle/macos/LocalScribe.app
#   6. 提示用户启动并配置 DeepSeek API Key
#
# 用法:
#   chmod +x install.sh && ./install.sh
#
# 可选环境变量:
#   SKIP_MODEL=1     不下模型(已下过)
#   SKIP_BUILD=1     不构建 .app(只装依赖)
#   HF_MIRROR=1      用 hf-mirror.com 国内加速
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

c_blue='\033[1;34m'; c_green='\033[1;32m'; c_yellow='\033[1;33m'; c_red='\033[1;31m'; c_reset='\033[0m'
step() { echo -e "${c_blue}▸ $*${c_reset}"; }
ok()   { echo -e "${c_green}✓ $*${c_reset}"; }
warn() { echo -e "${c_yellow}⚠ $*${c_reset}"; }
err()  { echo -e "${c_red}✕ $*${c_reset}"; }

# =============================================================================
# 0. 系统检查
# =============================================================================
if [[ "$(uname)" != "Darwin" ]] || [[ "$(uname -m)" != "arm64" ]]; then
  warn "当前脚本只在 macOS Apple Silicon (M1/M2/M3/M4) 上测试过。"
  warn "Intel Mac / Linux / Windows 需要手动跑相同步骤,faster-whisper 路径。"
fi

step "1/6 检查系统依赖"

need_brew=0
have() { command -v "$1" >/dev/null 2>&1; }

if ! have ffmpeg; then
  warn "ffmpeg 未安装,稍后将通过 brew 装"; need_brew=1
fi
if ! have uv; then
  warn "uv 未安装(Python 包管理器),稍后用 curl 装"
fi
if ! have pnpm; then
  warn "pnpm 未安装,稍后通过 brew 装"; need_brew=1
fi
if ! have cargo; then
  warn "Rust 未安装,稍后用官方 rustup 装"
fi

if [[ $need_brew -eq 1 ]]; then
  if ! have brew; then
    err "需要 ffmpeg / pnpm,但 Homebrew 未安装。请先访问 https://brew.sh 安装,然后重跑。"
    exit 1
  fi
  if ! have ffmpeg; then step "  brew install ffmpeg"; brew install ffmpeg; fi
  if ! have pnpm;   then step "  brew install pnpm";   brew install pnpm;   fi
fi

if ! have uv; then
  step "  装 uv (curl)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! have cargo; then
  step "  装 Rust (rustup)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  source "$HOME/.cargo/env"
fi
ok "系统依赖就绪 · ffmpeg / uv / pnpm / cargo 全部可用"

# =============================================================================
# 2. Python venv + sidecar
# =============================================================================
step "2/6 创建 Python venv 并装 scribe-py"
if [[ ! -d ".venv" ]]; then
  uv venv --python 3.12
fi

INDEX_URL_ARG=""
if [[ "${HF_MIRROR:-0}" == "1" ]] || [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 https://pypi.org)" != "200" ]]; then
  INDEX_URL_ARG="--index-url https://pypi.tuna.tsinghua.edu.cn/simple"
  step "  使用清华镜像"
fi

uv pip install --python .venv/bin/python $INDEX_URL_ARG -e scribe-py
uv pip install --python .venv/bin/python $INDEX_URL_ARG silero-vad
ok "Python sidecar + 依赖装好"

# =============================================================================
# 3. 下载 Whisper 模型(1.5 GB)
# =============================================================================
if [[ "${SKIP_MODEL:-0}" == "1" ]]; then
  warn "SKIP_MODEL=1,跳过模型下载"
else
  step "3/6 下载 Whisper 模型(1.5 GB,首次需要)"
  MODEL_DIR="$REPO_ROOT/models/whisper-large-v3-turbo"
  if [[ -f "$MODEL_DIR/weights.safetensors" ]]; then
    ok "模型已存在 ($MODEL_DIR),跳过"
  else
    mkdir -p "$MODEL_DIR"
    HF_ENV=""
    if [[ "${HF_MIRROR:-0}" == "1" ]]; then
      HF_ENV="HF_ENDPOINT=https://hf-mirror.com"
    fi
    step "  目标目录: $MODEL_DIR"
    eval $HF_ENV .venv/bin/python3 -c "
from huggingface_hub import snapshot_download
import os, shutil
print('downloading… (国内首次约 5-15 分钟)')
p = snapshot_download(
    repo_id='mlx-community/whisper-large-v3-turbo',
    local_dir='$MODEL_DIR',
    local_dir_use_symlinks=False,
)
print('done:', p)
"
    ok "模型下载完成 → $MODEL_DIR"
  fi
fi

# =============================================================================
# 4. 前端依赖
# =============================================================================
step "4/6 装前端依赖(pnpm)"
PNPM_REGISTRY=""
if [[ "${HF_MIRROR:-0}" == "1" ]] || [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 https://registry.npmjs.org)" != "200" ]]; then
  PNPM_REGISTRY="--registry=https://registry.npmmirror.com"
  step "  使用 npmmirror"
fi
pnpm install $PNPM_REGISTRY
ok "前端依赖装好"

# =============================================================================
# 5. 构建 .app
# =============================================================================
if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
  warn "SKIP_BUILD=1,跳过 .app 构建"
else
  step "5/6 构建 .app(release 优化,首次约 3-5 分钟)"
  pnpm tauri build
  APP="$REPO_ROOT/src-tauri/target/release/bundle/macos/LocalScribe.app"
  DMG=$(ls "$REPO_ROOT/src-tauri/target/release/bundle/dmg/"*.dmg 2>/dev/null | head -1)
  ok "构建完成"
  echo
  echo "    .app: $APP"
  [[ -n "$DMG" ]] && echo "    .dmg: $DMG"
fi

# =============================================================================
# 6. 后续步骤
# =============================================================================
echo
echo -e "${c_green}════════════════════════════════════════════════════════════${c_reset}"
echo -e "${c_green}  LocalScribe 安装完成!${c_reset}"
echo -e "${c_green}════════════════════════════════════════════════════════════${c_reset}"
echo
echo "下一步:"
echo
echo "  ${c_blue}1.${c_reset} 启动 LocalScribe.app:"
echo "       open $REPO_ROOT/src-tauri/target/release/bundle/macos/LocalScribe.app"
echo "     或拖到 Applications 后双击启动。"
echo
echo "  ${c_blue}2.${c_reset} 配置 LLM 校对(可选):"
echo "       打开应用 → ⚙ 设置 → 校对 → 启用 LLM 校对"
echo "       粘贴 DeepSeek API Key(申请:https://platform.deepseek.com)"
echo
echo "  ${c_blue}3.${c_reset} CLI 调用(可选,for AI agents):"
echo "       ln -s $REPO_ROOT/bin/localscribe /usr/local/bin/localscribe"
echo "       localscribe pipeline audio.m4a --json"
echo
echo "更多:README.md · CLI.md"
