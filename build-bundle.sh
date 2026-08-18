#!/usr/bin/env bash
# =============================================================================
# LocalScribe · 自包含 .app 打包 — staging 准备脚本
# =============================================================================
# 产出 src-tauri/bundle-staging/ 含:
#   python/                  # python-build-standalone 3.12 (~50 MB)
#     bin/python3            # 可独立运行的 Python
#     lib/python3.12/site-packages/  # SenseVoice/FunASR, mlx-whisper, silero-vad, openai…
#   scribe-py/               # 我们的代码 (复制源码,site-packages 里同时装为 editable→无需,直接 copy 源)
#   models/                  # 可选 Whisper 回退权重
#   modelscope/hub/models/   # SenseVoice + timing/VAD + CAM++ 声纹模型缓存
#   huggingface/hub/         # Qwen3-ASR 高风险片段离线复核模型缓存
#   huggingface/offline-asr-manifest.json  # 模型尺寸 + SHA-256 完整性清单
#   deepfilternet/           # DeepFilterNet3 AI 降噪模型缓存
#   bin/ffmpeg               # 静态 ffmpeg (arm64)
#
# 然后 tauri.conf.json 的 bundle.resources 把 staging/ 整个映射进
# .app/Contents/Resources/, run() 启动时探测到这些路径就走打包模式。
#
# 用法:  ./build-bundle.sh [--skip-python] [--skip-ffmpeg] [--skip-model]
# =============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

c_blue='\033[1;34m'; c_green='\033[1;32m'; c_yellow='\033[1;33m'; c_red='\033[1;31m'; c_reset='\033[0m'
step() { echo -e "${c_blue}▸ $*${c_reset}"; }
ok()   { echo -e "${c_green}✓ $*${c_reset}"; }
warn() { echo -e "${c_yellow}⚠ $*${c_reset}"; }
err()  { echo -e "${c_red}✕ $*${c_reset}"; }

has_arm64_slice() {
  local bin="$1"
  [[ -x "$bin" ]] || return 1
  lipo -archs "$bin" 2>/dev/null | grep -qw "arm64"
}

if [[ "$(uname)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  err "本脚本只支持 macOS Apple Silicon"; exit 1
fi

STAGING="$REPO_ROOT/src-tauri/bundle-staging"
mkdir -p "$STAGING"

SKIP_PYTHON=0; SKIP_FFMPEG=0; SKIP_MODEL=0
for arg in "$@"; do
  case $arg in
    --skip-python) SKIP_PYTHON=1;;
    --skip-ffmpeg) SKIP_FFMPEG=1;;
    --skip-model)  SKIP_MODEL=1;;
  esac
done

# =============================================================================
# 1. python-build-standalone (可重定位 Python 3.12,无系统依赖)
# =============================================================================
PY_VER="3.12.7"
PY_RELEASE="20241016"
PY_TGZ_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PY_RELEASE}/cpython-${PY_VER}+${PY_RELEASE}-aarch64-apple-darwin-install_only.tar.gz"

if [[ $SKIP_PYTHON -eq 0 ]]; then
  step "1/4 准备 python-build-standalone (${PY_VER})"
  PY_DIR="$STAGING/python"
  if [[ -x "$PY_DIR/bin/python3" ]]; then
    ok "  Python 已存在,跳过 (使用 --skip-python 跳过)"
  else
    rm -rf "$PY_DIR"
    TMP="$STAGING/python.tar.gz"
    step "  下载 $PY_TGZ_URL"
    curl -L --fail -o "$TMP" "$PY_TGZ_URL"
    step "  解压"
    tar -xzf "$TMP" -C "$STAGING"   # 解压成 python/
    rm "$TMP"
    ok "  Python 就绪 → $PY_DIR/bin/python3"
  fi

  step "  装 sidecar 依赖到打包 Python"
  BUNDLED_PY="$PY_DIR/bin/python3"
  SENKO_SDKROOT=""
  for sdk in \
    /Library/Developer/CommandLineTools/SDKs/MacOSX15.5.sdk \
    /Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk \
    /Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk \
    /Library/Developer/CommandLineTools/SDKs/MacOSX15.sdk \
    /Library/Developer/CommandLineTools/SDKs/MacOSX14.5.sdk \
    /Library/Developer/CommandLineTools/SDKs/MacOSX14.sdk
  do
    if [[ -d "$sdk" ]]; then
      SENKO_SDKROOT="$sdk"
      break
    fi
  done
  PIP_ENV=()
  if [[ -n "$SENKO_SDKROOT" ]]; then
    PIP_ENV=(env SDKROOT="$SENKO_SDKROOT" MACOSX_DEPLOYMENT_TARGET=14.0)
  fi

  # 用我们仓库里 .venv 已验证的同一组依赖
  "$BUNDLED_PY" -m pip install --upgrade pip
  "${PIP_ENV[@]}" "$BUNDLED_PY" -m pip install \
    -e "$REPO_ROOT/scribe-py" \
    silero-vad \
    "mlx-audio==0.3.1" \
    "funasr==1.3.14" \
    "modelscope[framework]==1.37.1"
  if [[ "${LOCALSCRIBE_SKIP_DEEPFILTERNET:-0}" != "1" ]]; then
    step "  尝试安装 DeepFilterNet AI 降噪(可选)"
    if "$BUNDLED_PY" -m pip install "deepfilternet>=0.5.6"; then
      DF_MODEL_OUTPUT="$(
        PYTHONPATH="$REPO_ROOT/scribe-py/src" "$BUNDLED_PY" - <<'PY'
from scribe_py.core.audio import deepfilter_available, ensure_deepfilter_model_dir, resolve_deepfilter_model_dir

model_dir = ensure_deepfilter_model_dir(download=True)
if model_dir is None:
    raise SystemExit("DeepFilterNet model unavailable")
print(f"MODEL_DIR={model_dir}")
print(f"AVAILABLE={deepfilter_available()}")
print(f"RESOLVED={resolve_deepfilter_model_dir()}")
PY
      )" || DF_MODEL_OUTPUT=""
      if [[ "$DF_MODEL_OUTPUT" == *"MODEL_DIR="* ]]; then
        DF_MODEL_DIR="$(printf '%s\n' "$DF_MODEL_OUTPUT" | awk -F= '/^MODEL_DIR=/{print $2; exit}')"
        DF_STAGING_DIR="$STAGING/deepfilternet/DeepFilterNet3"
        if [[ "$(cd "$DF_MODEL_DIR" 2>/dev/null && pwd -P)" == "$(cd "$DF_STAGING_DIR" 2>/dev/null && pwd -P)" ]]; then
          ok "  DeepFilterNet 模型已在 staging,无需自复制"
        elif [[ -f "$DF_MODEL_DIR/config.ini" && -d "$DF_MODEL_DIR/checkpoints" ]]; then
          rm -rf "$STAGING/deepfilternet"
          mkdir -p "$STAGING/deepfilternet"
          rsync -a --delete --exclude 'enhance.log' "$DF_MODEL_DIR/" "$DF_STAGING_DIR/"
        else
          warn "  DeepFilterNet 模型目录无效: $DF_MODEL_DIR; App 会自动回退轻降噪"
          rm -rf "$STAGING/deepfilternet"
        fi
        if [[ -f "$DF_STAGING_DIR/config.ini" && -d "$DF_STAGING_DIR/checkpoints" ]]; then
          ok "  DeepFilterNet AI 降噪可用,模型已缓存 → $DF_STAGING_DIR"
        fi
      else
        warn "  DeepFilterNet 已安装但模型准备失败,App 会自动回退轻降噪"
      fi
    else
      warn "  DeepFilterNet 安装失败,App 会自动回退轻降噪"
    fi
  else
    warn "  已跳过 DeepFilterNet 安装(LOCALSCRIBE_SKIP_DEEPFILTERNET=1)"
  fi
  LLVMLITE_DYLIB="$PY_DIR/lib/python3.12/site-packages/llvmlite/binding/libllvmlite.dylib"
  if [[ ! -f "$LLVMLITE_DYLIB" ]]; then
    warn "  llvmlite 动态库缺失,强制恢复 arm64 wheel"
    "$BUNDLED_PY" -m pip install --force-reinstall --no-deps "llvmlite==0.47.0"
  fi
  "$BUNDLED_PY" - <<'PY'
import senko
print("senko import ok")
PY
  ok "  依赖装好"

  # editable install 写的是绝对路径,打包后路径变了会找不到。
  # 把 scribe-py 源码改为直接 copy 进 Resources,然后用 PYTHONPATH 指向它。
  # 这里去掉 editable 链接,改为纯 site-packages 安装。
  "$BUNDLED_PY" -m pip uninstall -y scribe-py 2>/dev/null || true
  "${PIP_ENV[@]}" "$BUNDLED_PY" -m pip install --no-deps "$REPO_ROOT/scribe-py"
  ok "  scribe-py 转为非 editable 安装(打包模式必需)"

  MLX_METALLIB="$PY_DIR/lib/python3.12/site-packages/mlx/lib/mlx.metallib"
  if [[ ! -f "$MLX_METALLIB" ]]; then
    warn "  MLX Metal 资源缺失,强制恢复 mlx-metal wheel"
    "$BUNDLED_PY" -m pip install --force-reinstall --no-deps "mlx-metal==0.31.2"
  fi
  if ! "$BUNDLED_PY" -c "import torch" >/dev/null 2>&1; then
    warn "  PyTorch 动态库不完整,强制恢复 torch wheel"
    "$BUNDLED_PY" -m pip install --force-reinstall --no-deps "torch==2.12.0"
  fi

  # 验证打包 Python 能 import 关键依赖
  step "  验证依赖"
  "$BUNDLED_PY" -c "import addict, mlx_whisper, silero_vad, funasr, modelscope, openai, zhconv, scribe_py; print('OK openai=' + openai.__version__)"
fi

BUNDLED_PY="$STAGING/python/bin/python3"
if [[ -x "$BUNDLED_PY" ]]; then
  step "1/4 验证打包 Python 必需运行时依赖"
  if ! "$BUNDLED_PY" - <<'PY'
import funasr
import mlx_audio
import modelscope
import addict
import zhconv
from mlx_audio.stt.utils import load_model
print("asr runtime imports ok (including mlx-audio)")
PY
  then
    err "  必需 ASR 运行时缺失(SenseVoice 或 mlx-audio)"
    if [[ $SKIP_PYTHON -eq 1 ]]; then
      err "  当前使用了 --skip-python;请去掉该参数重建打包 Python"
    fi
    exit 1
  fi
  "$BUNDLED_PY" - <<'PY'
from zhconv import convert
assert convert("當我們進入學校的時候", "zh-hans") == "当我们进入学校的时候"
print("simplified conversion ok")
PY

  # Python 会在导入业务模块前自动执行 sitecustomize。它只在完整清单和
  # mlx-audio 同时存在时暴露 App 内 HF 缓存,否则固定指向空缓存并保持离线。
  SITE_PACKAGES="$("$BUNDLED_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  cp "$REPO_ROOT/scripts/localscribe_bundled_sitecustomize.py" "$SITE_PACKAGES/sitecustomize.py"
  ok "  离线 Qwen 启动配置 → $SITE_PACKAGES/sitecustomize.py"
else
  err "  打包 Python 不存在: $BUNDLED_PY"
  exit 1
fi

# =============================================================================
# 2. scribe-py 源码
# =============================================================================
step "2/4 复制 scribe-py 源码到 staging"
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' \
  "$REPO_ROOT/scribe-py/" "$STAGING/scribe-py/"
ok "  scribe-py 同步完成"

# The customer App never runs package test suites.  Strip only conventional
# test folders and bytecode caches; do not touch runtime modules such as
# torch.testing, model examples, dynamic libraries, metadata, or licenses.
step "2/4 清理打包运行时测试文件"
SITE_PACKAGES="$STAGING/python/lib/python3.12/site-packages"
if [[ -d "$SITE_PACKAGES" ]]; then
  find "$SITE_PACKAGES" -type d \( -name tests -o -name test \) -prune -exec rm -rf {} +
  find "$SITE_PACKAGES" -type d -name '__pycache__' -prune -exec rm -rf {} +
  find "$SITE_PACKAGES" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
fi
rm -rf "$STAGING/scribe-py/tests"
ok "  已移除运行时不需要的测试代码与 Python 缓存"

# =============================================================================
# 3. ffmpeg 静态二进制 (macOS Apple Silicon arm64)
# =============================================================================
if [[ $SKIP_FFMPEG -eq 0 ]]; then
  step "3/4 准备 ffmpeg 静态二进制"
  FF_DIR="$STAGING/bin"
  mkdir -p "$FF_DIR"
  FF_BASE_URL="https://ffmpeg.martin-riedl.de/redirect/latest/macos/arm64/release"

  install_ff_tool() {
    local name="$1"
    local url="$2"
    local dest="$FF_DIR/$name"
    if has_arm64_slice "$dest"; then
      ok "  $name 已存在且为 arm64,跳过"
      return
    fi
    if [[ -e "$dest" ]]; then
      warn "  $name 不是 arm64,重新下载"
      rm -f "$dest"
    fi
    local tmp="$STAGING/${name}.zip"
    step "  下载 $name (macOS arm64)"
    curl -L --fail -o "$tmp" "$url"
    unzip -o "$tmp" -d "$FF_DIR" >/dev/null
    rm "$tmp"
    chmod +x "$dest"
    if ! has_arm64_slice "$dest"; then
      file "$dest" || true
      err "  $name 不是 arm64,中止"
      exit 1
    fi
    ok "  $name → $dest ($(lipo -archs "$dest"))"
  }

  install_ff_tool "ffmpeg" "$FF_BASE_URL/ffmpeg.zip"
  # ffprobe 也来一份(scribe-py 的 audio.py 用)
  install_ff_tool "ffprobe" "$FF_BASE_URL/ffprobe.zip"
fi

# =============================================================================
# 4. 可选 Whisper 回退模型 + 必需 SenseVoice/Paraformer/Qwen
# =============================================================================
if [[ $SKIP_MODEL -eq 0 ]]; then
  step "4/4 检查可选 Whisper 回退模型"
  SRC="$REPO_ROOT/models/whisper-large-v3-turbo"
  DST="$STAGING/models/whisper-large-v3-turbo"
  mkdir -p "$STAGING/models"
  if [[ -s "$SRC/weights.safetensors" ]]; then
    mkdir -p "$DST"
    # APFS clonefile = 秒拷
    cp -c "$SRC/weights.safetensors" "$DST/" 2>/dev/null || cp "$SRC/weights.safetensors" "$DST/"
    cp "$SRC/config.json" "$DST/"
    [[ -f "$SRC/README.md" ]] && cp "$SRC/README.md" "$DST/"
    ok "  Whisper 回退模型 → $DST"
  else
    rm -rf "$DST"
    warn "  未提供完整 Whisper 权重；客户包将使用必需的 SenseVoice 主转录和本地 Qwen 复核"
  fi

  step "4/4 复制 SenseVoice / FunASR 模型到 staging"
  MS_SRC_ROOT="${MODELSCOPE_CACHE:-$HOME/.cache/modelscope/hub}/models"
  MS_DST_ROOT="$STAGING/modelscope/hub/models"

  copy_modelscope_model() {
    local repo="$1"
    local src="$MS_SRC_ROOT/$repo"
    local dst="$MS_DST_ROOT/$repo"
    if [[ ! -f "$src/config.yaml" && ! -f "$src/configuration.json" ]]; then
      err "  ModelScope 模型不存在: $src"
      err "  先在开发环境跑一次 SenseVoice,或用 modelscope 下载 $repo"
      exit 1
    fi
    mkdir -p "$(dirname "$dst")"
    rm -rf "$dst"
    if ! cp -cR "$src" "$dst" 2>/dev/null; then
      rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' "$src/" "$dst/"
    fi
    find "$dst" -type d -name '__pycache__' -prune -exec rm -rf {} +
    find "$dst" -type f -name '*.pyc' -delete
    ok "  $repo → $dst"
  }

  copy_modelscope_model "iic/SenseVoiceSmall"
  copy_modelscope_model "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
  copy_modelscope_model "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
  copy_modelscope_model "damo/speech_campplus_sv_zh-cn_16k-common"

  step "4/4 复制 Qwen3-ASR 高风险片段复核模型到 staging"
  if [[ -n "${HF_HUB_CACHE:-}" ]]; then
    HF_SRC_ROOT="$HF_HUB_CACHE"
  elif [[ -n "${HUGGINGFACE_HUB_CACHE:-}" ]]; then
    HF_SRC_ROOT="$HUGGINGFACE_HUB_CACHE"
  elif [[ -n "${HF_HOME:-}" ]]; then
    HF_SRC_ROOT="$HF_HOME/hub"
  else
    HF_SRC_ROOT="$HOME/.cache/huggingface/hub"
  fi
  QWEN_CACHE_NAME="models--mlx-community--Qwen3-ASR-1.7B-8bit"
  QWEN_SRC="$HF_SRC_ROOT/$QWEN_CACHE_NAME"
  QWEN_DST="$STAGING/huggingface/hub/$QWEN_CACHE_NAME"
  if [[ ! -f "$QWEN_SRC/refs/main" ]]; then
    err "  Qwen3-ASR 本地缓存不存在或不完整: $QWEN_SRC"
    err "  客户包禁止运行时下载;请先在构建机准备 mlx-community/Qwen3-ASR-1.7B-8bit"
    exit 1
  fi
  mkdir -p "$(dirname "$QWEN_DST")"
  rm -rf "$QWEN_DST"
  if ! cp -cR "$QWEN_SRC" "$QWEN_DST" 2>/dev/null; then
    rsync -a --delete \
      --exclude '.locks' --exclude '*.incomplete' \
      "$QWEN_SRC/" "$QWEN_DST/"
  fi
  find "$QWEN_DST" -type f -name '*.incomplete' -delete
  python3 "$REPO_ROOT/scripts/offline_asr_bundle.py" write-manifest --resources "$STAGING"
  ok "  Qwen3-ASR → $QWEN_DST ($(du -sh "$QWEN_DST" | cut -f1))"

  step "4/4 验证打包 SenseVoice 后端选择"
  LOCALSCRIBE_MODELSCOPE_CACHE="$STAGING/modelscope/hub" \
  MODELSCOPE_CACHE="$STAGING/modelscope/hub" \
  PYTHONPATH="$STAGING/scribe-py/src" \
    "$BUNDLED_PY" - <<'PY'
from scribe_py.core.selector import default_backend, default_model_id
from scribe_py.core.transcriber_funasr import DEFAULT_MODEL, model_cached, SENSEVOICE_MODEL
from scribe_py.diarizers.exact_embedding_fallback import resolve_local_model_path
import os
from pathlib import Path

assert model_cached(SENSEVOICE_MODEL), "bundled SenseVoice model cache is missing"
assert model_cached(DEFAULT_MODEL), "bundled Paraformer timing model cache is missing"
speaker_model = resolve_local_model_path()
staging_modelscope = Path(os.environ["LOCALSCRIBE_MODELSCOPE_CACHE"]).resolve()
assert speaker_model.is_relative_to(staging_modelscope), (
    f"bundled CAM++ speaker model resolved outside staging: {speaker_model}"
)
assert default_backend() == "sensevoice", f"default_backend={default_backend()!r}"
assert default_model_id("auto") == SENSEVOICE_MODEL
print(f"bundled SenseVoice backend and CAM++ speaker fallback ok: {speaker_model}")
PY
fi

step "4/4 离线 Qwen3-ASR 打包 preflight"
python3 "$REPO_ROOT/scripts/offline_asr_bundle.py" verify \
  --resources "$STAGING" --check-hash
LOCALSCRIBE_RESOURCES="$STAGING" \
  "$BUNDLED_PY" - <<'PY'
import os
from huggingface_hub import try_to_load_from_cache
from mlx_audio.stt.utils import load_model

model_id = "mlx-community/Qwen3-ASR-1.7B-8bit"
assert os.environ.get("HF_HUB_OFFLINE") == "1", "bundled Hugging Face runtime is not offline"
assert os.environ.get("TRANSFORMERS_OFFLINE") == "1", "bundled Transformers runtime is not offline"
assert os.environ.get("LOCALSCRIBE_OFFLINE_QWEN_AVAILABLE") == "1", (
    "bundled Qwen unavailable: " + os.environ.get("LOCALSCRIBE_OFFLINE_QWEN_REASON", "unknown")
)
config = try_to_load_from_cache(model_id, "config.json")
assert isinstance(config, str), "bundled Qwen config is not discoverable from the offline cache"
print("bundled offline Qwen cache and mlx-audio runtime ok")
PY

# =============================================================================
# 总结
# =============================================================================
echo
ok "Staging 完成 → $STAGING"
du -sh "$STAGING"/* 2>/dev/null
echo
echo "下一步:  pnpm tauri build"
echo "        (tauri.conf.json 的 bundle.resources 会把 staging/ 拷进 .app/Contents/Resources/)"
