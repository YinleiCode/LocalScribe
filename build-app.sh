#!/usr/bin/env bash
# =============================================================================
# LocalScribe · 自包含 .app + .dmg 完整打包
# =============================================================================
# 1. ./build-bundle.sh                  → src-tauri/bundle-staging/ 准备好资源
# 2. pnpm tauri build                   → 出基础 .app + .dmg (不含资源)
# 3. 手动 rsync staging/ → .app/Contents/Resources/  (保留 symlink + exec bit)
# 4. 重新生成 .dmg
#
# 用法:  ./build-app.sh [--skip-staging] [--app-only]
# =============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

c_blue='\033[1;34m'; c_green='\033[1;32m'; c_yellow='\033[1;33m'; c_red='\033[1;31m'; c_reset='\033[0m'
step() { echo -e "${c_blue}▸ $*${c_reset}"; }
ok()   { echo -e "${c_green}✓ $*${c_reset}"; }
warn() { echo -e "${c_yellow}⚠ $*${c_reset}"; }
err()  { echo -e "${c_red}✕ $*${c_reset}"; }

rust_toolchain_is_compatible() {
  local bin_dir="$1"
  local version major minor
  [[ -x "$bin_dir/cargo" && -x "$bin_dir/rustc" ]] || return 1
  version="$($bin_dir/rustc --version 2>/dev/null | awk '{print $2}')"
  IFS=. read -r major minor _ <<< "$version"
  [[ "$major" =~ ^[0-9]+$ && "$minor" =~ ^[0-9]+$ ]] || return 1
  (( major > 1 || (major == 1 && minor >= 88) ))
}

select_rust_toolchain() {
  local candidate current_cargo
  current_cargo="$(command -v cargo 2>/dev/null || true)"
  if [[ -n "$current_cargo" ]]; then
    candidate="$(dirname "$current_cargo")"
    if rust_toolchain_is_compatible "$candidate"; then
      export PATH="$candidate:$PATH"
      ok "  Rust 工具链: $(rustc --version)"
      return
    fi
  fi

  for candidate in \
    "$HOME"/.rustup/toolchains/stable-*/bin \
    /opt/homebrew/bin \
    /usr/local/bin
  do
    if rust_toolchain_is_compatible "$candidate"; then
      export PATH="$candidate:$PATH"
      ok "  Rust 工具链: $(rustc --version)"
      return
    fi
  done

  err "需要 Rust 1.88 或更高版本才能构建当前锁定依赖"
  exit 1
}

SKIP_STAGING=0
APP_ONLY=0
for arg in "$@"; do
  case $arg in
    --skip-staging) SKIP_STAGING=1;;
    --app-only) APP_ONLY=1;;
  esac
done

STAGING="$REPO_ROOT/src-tauri/bundle-staging"

select_rust_toolchain

verify_offline_asr_bundle() {
  local resources="$1"
  local python_bin="$2"
  local load_model="${3:-0}"

  python3 "$REPO_ROOT/scripts/offline_asr_bundle.py" verify \
    --resources "$resources" --check-hash
  if [[ "${LOCALSCRIBE_SKIP_GPU_MODEL_PRECHECK:-0}" == "1" ]]; then
    warn "  已跳过 GPU 模型实加载预检（仅限无 Metal 的构建环境）"
    return
  fi
  LOCALSCRIBE_RESOURCES="$resources" \
    "$python_bin" - "$load_model" <<'PY'
import os
import sys
from huggingface_hub import try_to_load_from_cache
from mlx_audio.stt.utils import load_model

model_id = "mlx-community/Qwen3-ASR-1.7B-8bit"
assert os.environ.get("HF_HUB_OFFLINE") == "1", "Hugging Face runtime is not offline"
assert os.environ.get("TRANSFORMERS_OFFLINE") == "1", "Transformers runtime is not offline"
assert os.environ.get("LOCALSCRIBE_OFFLINE_QWEN_AVAILABLE") == "1", (
    "offline Qwen unavailable: " + os.environ.get("LOCALSCRIBE_OFFLINE_QWEN_REASON", "unknown")
)
config = try_to_load_from_cache(model_id, "config.json")
assert isinstance(config, str), "offline Qwen config is not discoverable"
if sys.argv[1] == "1":
    model = load_model(model_id)
    assert model is not None, "mlx-audio returned no Qwen model"
    del model
print("offline Qwen preflight ok; full_load=" + sys.argv[1])
PY
}

verify_campp_speaker_bundle() {
  local resources="$1"
  local python_bin="$2"

  if [[ "${LOCALSCRIBE_SKIP_GPU_MODEL_PRECHECK:-0}" == "1" ]]; then
    warn "  已跳过 CAM++ GPU 模型实加载预检（仅限无 Metal 的构建环境）"
    return
  fi

  LOCALSCRIBE_RESOURCES="$resources" \
  LOCALSCRIBE_MODELSCOPE_CACHE="$resources/modelscope/hub" \
  MODELSCOPE_CACHE="$resources/modelscope/hub" \
  PYTHONPATH="$resources/scribe-py/src" \
    "$python_bin" - <<'PY'
import os
import wave
from pathlib import Path

import numpy as np
import torch
from modelscope.pipelines import pipeline
from senko import config as senko_config
from senko.vad_local_pyannote import LocalSegmentationVADCuda

from scribe_py.diarizers.exact_embedding_fallback import resolve_local_model_path

resources = Path(os.environ["LOCALSCRIBE_RESOURCES"]).resolve()
model = resolve_local_model_path()
assert model.is_relative_to(resources / "modelscope/hub"), (
    f"CAM++ speaker model resolved outside packaged resources: {model}"
)
assert (model / "campplus_cn_common.bin").stat().st_size > 1_000_000, (
    f"CAM++ speaker model weights are incomplete: {model}"
)
example = model / "examples/speaker1_a_cn_16k.wav"
assert example.is_file(), f"CAM++ speaker model example is missing: {example}"
with wave.open(str(example), "rb") as wav:
    assert wav.getframerate() == 16_000 and wav.getnchannels() == 1
    samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32)
samples /= 32768.0
os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
speaker_pipeline = pipeline(task="speaker-verification", model=str(model))
result = speaker_pipeline([samples], output_emb=True)
embeddings = np.asarray(result.get("embs"), dtype=np.float32)
if embeddings.ndim == 1:
    embeddings = embeddings[None, :]
assert embeddings.ndim == 2 and embeddings.shape[0] == 1 and embeddings.shape[1] > 0
assert np.all(np.isfinite(embeddings)), "CAM++ speaker model produced non-finite embeddings"
senko_paths = senko_config.resolve_model_paths(
    required_fields=("pyannote_segmentation_senko_model_path",),
)
vad_checkpoint = Path(senko_paths.pyannote_segmentation_senko_model_path).resolve()
assert vad_checkpoint.is_relative_to(resources / "python"), (
    f"Senko PyTorch VAD resolved outside packaged resources: {vad_checkpoint}"
)
vad = LocalSegmentationVADCuda(
    checkpoint_path=vad_checkpoint,
    torch_device=torch.device("cpu"),
)
assert vad.device.type == "cpu" and vad.sample_rate == 16_000
print(
    f"CAM++ and PyTorch VAD fallback preflight ok: {model}; "
    f"dim={embeddings.shape[1]}; vad={vad_checkpoint}"
)
PY
}

# =============================================================================
# 1. 准备 staging
# =============================================================================
if [[ $SKIP_STAGING -eq 0 ]]; then
  step "1/4 准备 staging (Python + 依赖 + ffmpeg + 模型)"
  ./build-bundle.sh
else
  step "1/4 跳过 staging (--skip-staging)"
  if [[ ! -x "$STAGING/python/bin/python3" ]]; then
    err "  staging 不完整 ($STAGING),不能跳过"
    exit 1
  fi
fi

step "  校验 staging 内离线 Qwen 模型与 mlx-audio"
verify_offline_asr_bundle "$STAGING" "$STAGING/python/bin/python3" 0
step "  校验 staging 内 CAM++ 缺失证据补偿模型"
verify_campp_speaker_bundle "$STAGING" "$STAGING/python/bin/python3"

# =============================================================================
# 2. tauri build (基础 .app,不含资源)
# =============================================================================
step "2/4 pnpm tauri build"

# ─── 卸载所有 LocalScribe 相关挂载点 ────────────────────────────────────────
# 经验:Tauri 的 bundle_dmg.sh 在挂载点冲突时会静默失败,然后 .app 不被注入资源
# 但 build-app.sh 仍然 exit 0(set -e 没生效,原因不明)。所以这里必须**铁腕清理**:
#   1. 按卷名 /Volumes/LocalScribe 找挂载点(任何分支版本,如 -1, -2)
#   2. 按 image-path 含 LocalScribe 的 image 全部 detach
#   3. 删 Tauri 临时 rw.*.dmg
warn "  清理任何已挂载的 LocalScribe 卷 / 旧 DMG"
mount | awk '/\/Volumes\/LocalScribe/ {print $1}' | while read dev; do
  hdiutil detach "$dev" -force 2>/dev/null && echo "    ✓ detached $dev"
done
# hdiutil 列出所有 image,把含 LocalScribe 的 dev node detach
hdiutil info -plist 2>/dev/null \
  | python3 -c "
import sys, plistlib
try:
    d = plistlib.loads(sys.stdin.buffer.read())
    for img in d.get('images', []):
        path = img.get('image-path', '')
        if 'LocalScribe' not in path: continue
        for entity in img.get('system-entities', []):
            dev = entity.get('dev-entry')
            if dev: print(dev)
except Exception:
    pass
" | sort -u | while read dev; do
  hdiutil detach "$dev" -force 2>/dev/null && echo "    ✓ detached image: $dev"
done
rm -f "$REPO_ROOT/src-tauri/target/release/bundle/macos/rw."*.dmg 2>/dev/null || true
rm -f "$REPO_ROOT/src-tauri/target/release/bundle/dmg/rw."*.dmg 2>/dev/null || true

# Tauri 在 DMG bundle 步骤失败时,前面的 .app 步骤其实成功了,但 pnpm 会以 exit 1 收尾。
# 我们只需要 .app(自己重打 DMG),所以容忍 tauri 的 DMG 失败,只要 .app 还在就继续。
pnpm tauri build --bundles app || warn "  tauri 退出非零，但 .app 通常已生成，继续检查产物"

APP="$REPO_ROOT/src-tauri/target/release/bundle/macos/LocalScribe.app"
DMG_DIR="$REPO_ROOT/src-tauri/target/release/bundle/dmg"
mkdir -p "$DMG_DIR"

if [[ ! -d "$APP" ]]; then
  err "tauri build 没产出 .app"; exit 1
fi
# 把 Tauri 留下的 rw 临时 dmg 再扫一次清掉(避免后续 hdiutil 再被挂载阻塞)
hdiutil info -plist 2>/dev/null | python3 -c "
import sys, plistlib
d = plistlib.loads(sys.stdin.buffer.read())
for img in d.get('images', []):
    if 'LocalScribe' in img.get('image-path',''):
        for ent in img.get('system-entities', []):
            dev = ent.get('dev-entry')
            if dev: print(dev)
" 2>/dev/null | sort -u | while read dev; do
  hdiutil detach "$dev" -force 2>/dev/null && echo "    ✓ post-tauri detach $dev"
done
rm -f "$REPO_ROOT/src-tauri/target/release/bundle/macos/rw."*.dmg 2>/dev/null || true

ok "  $APP"

# =============================================================================
# 3. 把 staging 内容塞进 .app/Contents/Resources/
# =============================================================================
step "3/4 注入 Python / scribe-py / models / ModelScope / Qwen3-ASR / DeepFilterNet / ffmpeg"
RES_DIR="$APP/Contents/Resources"
mkdir -p "$RES_DIR"

# 先清空旧残留,再用 ditto(macOS 原生工具,完美保留 symlinks/xattrs/perms)
# rsync -a 在 .app 下会触发 utimensat 权限错误;cp -R 不保留 symlinks。
# ditto 是 Apple 推荐的 .app 内复制方式。
# 关键: macOS Tahoe 不让把带 com.apple.provenance xattr 的文件放进 .app
# (该 xattr 来自下载的 python-build-standalone tarball)
# ditto/rsync/cp 都会触发 "Operation not permitted"
# 解决: 先一次性剥光 staging 的 xattr,再用 tar 管道复制(symlinks 自动保留)
step "  剥离 staging 的 xattr (com.apple.*)"
xattr -dr com.apple.provenance "$STAGING" 2>/dev/null || true
xattr -dr com.apple.quarantine  "$STAGING" 2>/dev/null || true

for sub in python scribe-py models modelscope huggingface bin deepfilternet; do
  if [[ ! -d "$STAGING/$sub" ]]; then
    if [[ "$sub" == "deepfilternet" ]]; then
      rm -rf "$RES_DIR/$sub"
      warn "  DeepFilterNet 模型未进入 staging,AI 降噪运行时会自动回退"
      continue
    fi
    err "  staging 缺少必要目录: $STAGING/$sub"
    exit 1
  fi
  rm -rf "$RES_DIR/$sub"
  step "  clone-copy $sub"
  if ! cp -cR "$STAGING/$sub" "$RES_DIR/$sub" 2>/dev/null; then
    mkdir -p "$RES_DIR/$sub"
    ( cd "$STAGING/$sub" && tar -cf - . ) | ( cd "$RES_DIR/$sub" && tar -xpf - )
  fi
done

# 确保 ffmpeg / ffprobe / python3 是可执行的
chmod +x "$RES_DIR/bin/ffmpeg" "$RES_DIR/bin/ffprobe" 2>/dev/null || true
chmod +x "$RES_DIR/python/bin/python3" 2>/dev/null || true

# 验证打包 Python 可以 import 关键依赖
step "  验证打包 Python 能 import 依赖"
LOCALSCRIBE_RESOURCES="$RES_DIR" \
LOCALSCRIBE_MODELSCOPE_CACHE="$RES_DIR/modelscope/hub" \
MODELSCOPE_CACHE="$RES_DIR/modelscope/hub" \
"$RES_DIR/python/bin/python3" -c "
import os, sys
sys.path = [p for p in sys.path if 'site-packages' not in p] + [p for p in sys.path if 'site-packages' in p]
sys.path.insert(0, r'$RES_DIR/scribe-py/src')
from scribe_py.core.selector import default_backend
from scribe_py.core.transcriber_funasr import DEFAULT_MODEL, SENSEVOICE_MODEL, model_cached
assert model_cached(SENSEVOICE_MODEL), 'SenseVoice model cache missing in app Resources'
assert model_cached(DEFAULT_MODEL), 'Paraformer timing/review model cache missing in app Resources'
assert default_backend() == 'sensevoice', 'default_backend is not sensevoice'
import mlx_audio, mlx_whisper, silero_vad, funasr, modelscope, openai, zhconv
try:
    from scribe_py.core.audio import deepfilter_available, resolve_deepfilter_model_dir
    model_dir = resolve_deepfilter_model_dir()
    deepfilter = 'deepfilternet=available(model=' + str(model_dir) + ')' if deepfilter_available() else 'deepfilternet=unavailable(model_missing)'
except Exception as exc:
    deepfilter = 'deepfilternet=unavailable(' + type(exc).__name__ + ')'
print('  ✓ mlx_audio, mlx_whisper, silero_vad, funasr, modelscope, zhconv, openai 全部可 import; default_backend=sensevoice; openai=' + openai.__version__ + '; ' + deepfilter)
"

step "  完整加载 App 内 Qwen3-ASR(强制离线)"
verify_offline_asr_bundle "$RES_DIR" "$RES_DIR/python/bin/python3" 1
step "  校验 App 内 CAM++ 缺失证据补偿模型"
verify_campp_speaker_bundle "$RES_DIR" "$RES_DIR/python/bin/python3"

APP_SIZE=$(du -sh "$APP" | cut -f1)
APP_KB=$(du -s "$APP" | cut -f1)
# 客户包额外包含约 2.3 GB Qwen3-ASR。模型/依赖完整性由上面的 preflight
# 精确校验;这里的 1 GB 下限继续防止生成完全没有资源的 Tauri 空壳。
# 不能拿这个空壳生成 DMG / 装到 /Applications 里(不然用户会得到 14 MB 不能跑的 .app)
if [[ $APP_KB -lt 1000000 ]]; then
  err "  .app 异常小($APP_SIZE),注入步骤一定失败了 — 中止"
  exit 1
fi
ok "  .app 注入完成,总大小 $APP_SIZE"

# 触发 macOS 重新签 quarantine 状态(去掉旧的)
xattr -dr com.apple.quarantine "$APP" 2>/dev/null || true

# Tauri 先签基础外壳，随后注入离线资源会使资源封印失效。客户本地包没有
# Developer ID 时使用 ad-hoc 重签，至少保证 macOS 对完整 App 的一致性校验通过。
step "  重新签名并验证注入后的完整 App"
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"
ok "  App 签名校验通过"

# =============================================================================
# 4. 重新生成 .dmg
# =============================================================================
if [[ $APP_ONLY -eq 1 ]]; then
  step "4/4 跳过 .dmg (--app-only)"
  echo
  echo -e "${c_green}═══════════════════════════════════════════════════════════${c_reset}"
  echo -e "${c_green}  App 构建完成!${c_reset}"
  echo -e "${c_green}═══════════════════════════════════════════════════════════${c_reset}"
  echo
  echo "  📦 .app:  $APP    ($APP_SIZE)"
  echo
  exit 0
fi

step "4/4 重新生成 .dmg"
APP_VERSION=$(awk -F'"' '/^[[:space:]]*"version"[[:space:]]*:/ {print $4; exit}' "$REPO_ROOT/src-tauri/tauri.conf.json")
if [[ -z "$APP_VERSION" ]]; then
  err "无法从 tauri.conf.json 读到版本号"; exit 1
fi
DMG_PATH="$DMG_DIR/LocalScribe_${APP_VERSION}_aarch64.dmg"

# 删掉旧 dmg 让 tauri 重新生成 — 但只跑 dmg 步骤太麻烦,直接 hdiutil 简单做
rm -f "$DMG_PATH"

# 生成 DMG (压缩,容量自动)
TMP_DIR=$(mktemp -d)
cleanup_dmg_tmp() {
  rm -rf "$TMP_DIR"
}
trap cleanup_dmg_tmp EXIT
mkdir -p "$TMP_DIR/dmg-source"
cp -cR "$APP" "$TMP_DIR/dmg-source/" 2>/dev/null || cp -R "$APP" "$TMP_DIR/dmg-source/"
ln -s /Applications "$TMP_DIR/dmg-source/Applications"

hdiutil create -volname "LocalScribe" \
  -srcfolder "$TMP_DIR/dmg-source" \
  -ov -format UDZO -fs HFS+ \
  "$DMG_PATH"
cleanup_dmg_tmp
trap - EXIT

DMG_SIZE=$(du -sh "$DMG_PATH" | cut -f1)
ok "  .dmg 完成: $DMG_PATH ($DMG_SIZE)"

# =============================================================================
echo
echo -e "${c_green}═══════════════════════════════════════════════════════════${c_reset}"
echo -e "${c_green}  完成!${c_reset}"
echo -e "${c_green}═══════════════════════════════════════════════════════════${c_reset}"
echo
echo "  📦 .app:  $APP    ($APP_SIZE)"
echo "  💿 .dmg:  $DMG_PATH    ($DMG_SIZE)"
echo
echo "测试:"
echo "  open '$DMG_PATH'      # 打开 dmg → 拖到 /Applications → 启动"
echo
