# LocalScribe 离线高风险 ASR 复核打包

## 发布约束

- 客户 App 不允许在运行时下载模型。
- 高风险复核使用 `mlx-community/Qwen3-ASR-1.7B-8bit` 和 `mlx-audio==0.3.1`。
- SenseVoice 仍是主转录后端；Qwen 只作为保守的本地复核证据。
- 构建机缓存不等于客户可用。模型、依赖和离线缓存定位都必须进入 `.app`。

## 构建机准备

构建前，Hugging Face hub 缓存中必须存在完整模型：

```text
~/.cache/huggingface/hub/
  models--mlx-community--Qwen3-ASR-1.7B-8bit/
    refs/main
    blobs/
    snapshots/<revision>/
```

可通过 `HF_HUB_CACHE`、`HUGGINGFACE_HUB_CACHE` 或 `HF_HOME` 指向其他构建缓存。
构建脚本只复制已有缓存，不负责下载 Qwen 模型。

## 构建和检查

```bash
./build-bundle.sh
pnpm package:preflight
./build-app.sh --skip-staging
```

`build-bundle.sh` 会：

1. 校验打包 Python 可以导入 `mlx_audio.stt.utils`。
2. 把 Hugging Face 模型缓存复制到 `Resources/huggingface/hub`。
3. 排除 `.incomplete` 文件并生成 `offline-asr-manifest.json`。
4. 对权重、配置、词表和索引写入尺寸及 SHA-256。
5. 安装 `sitecustomize.py`，使打包 Python 在业务代码导入前固定使用 App 内缓存并强制离线。

`build-app.sh` 在使用 staging 前和注入 App 后各校验一次清单。注入后还会在
`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 条件下用 `mlx-audio` 完整加载模型。
Whisper 只作为可选回退：存在完整 `weights.safetensors` 时一并打包；不存在时不会把
只有 `config.json` 的占位目录写入客户 App。SenseVoice、Paraformer、Qwen 和各自运行时
仍是硬门禁，任何一步失败都不会继续生成客户 DMG。

## 缺失和损坏时的行为

打包 Python 启动时会轻量核对清单中的文件和尺寸：

- 完整且 `mlx-audio` 存在：设置 `LOCALSCRIBE_OFFLINE_QWEN_AVAILABLE=1`，使用 App 内缓存。
- 缺失、尺寸异常或依赖缺失：设置为 `0`，把 Hugging Face 指向空的禁用缓存。
- 两种情况都强制 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。

因此损坏的安装不会尝试联网，也不会借用开发机或客户机器上偶然存在的模型缓存。
主转录会保留原始 SenseVoice 结果；官方发布包则必须通过构建 preflight，不能以降级状态出包。

## 体积

Qwen 权重为 `2,463,307,541` 字节，加上词表和配置后约 `2.47 GB` 十进制，磁盘显示约
`2.3 GiB`。`mlx-audio` 及其 Python 依赖还会增加少量体积；DMG 的实际增量取决于模型压缩率。

## 自动测试

```bash
pnpm test:package-offline-asr
```

测试覆盖完整清单、同尺寸内容损坏的 SHA-256 拒绝，以及缺文件时的离线安全降级。
