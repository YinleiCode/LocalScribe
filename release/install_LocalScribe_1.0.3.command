#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSET_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$ASSET_DIR/LocalScribe_1.0.3"
APP="$DEST/LocalScribe.app"

required=(
  "LocalScribe_1.0.3_app-base.tar.gz"
  "LocalScribe_1.0.3_python-runtime.tar.gz"
  "LocalScribe_1.0.3_modelscope.tar.gz"
  "LocalScribe_1.0.3_huggingface.tar.gz.part-aa"
  "LocalScribe_1.0.3_huggingface.tar.gz.part-ab"
)

for name in "${required[@]}"; do
  if [[ ! -f "$ASSET_DIR/$name" ]]; then
    echo "缺少发布组件: $name" >&2
    exit 1
  fi
done

if [[ -e "$DEST" ]]; then
  echo "目标目录已存在，请先移动或删除后重试: $DEST" >&2
  exit 1
fi

echo "[1/5] 校验下载文件"
(
  cd "$ASSET_DIR"
  shasum -a 256 -c "$SCRIPT_DIR/SHA256SUMS.txt"
)

mkdir -p "$DEST"
cleanup_on_error() {
  echo "安装失败，正在清理不完整输出: $DEST" >&2
  rm -rf "$DEST"
}
trap cleanup_on_error ERR

echo "[2/5] 还原 App 主体"
tar -xzf "$ASSET_DIR/LocalScribe_1.0.3_app-base.tar.gz" -C "$DEST"

echo "[3/5] 还原本地运行环境"
tar -xzf "$ASSET_DIR/LocalScribe_1.0.3_python-runtime.tar.gz" -C "$DEST"

echo "[4/5] 还原离线模型"
tar -xzf "$ASSET_DIR/LocalScribe_1.0.3_modelscope.tar.gz" -C "$DEST"
cat \
  "$ASSET_DIR/LocalScribe_1.0.3_huggingface.tar.gz.part-aa" \
  "$ASSET_DIR/LocalScribe_1.0.3_huggingface.tar.gz.part-ab" \
  | tar -xzf - -C "$DEST"

echo "[5/5] 验证 App 签名"
codesign --verify --deep --strict --verbose=2 "$APP"
trap - ERR

echo
echo "安装包还原完成: $APP"
echo "请将 LocalScribe.app 移到“应用程序”文件夹，首次启动时右键选择“打开”。"
open -R "$APP" 2>/dev/null || true
