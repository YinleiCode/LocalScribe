# scribe-py

LocalScribe 的 Python sidecar。封装 Whisper 转录 + LLM 校对/排版 流水线。

## 用法

### 作为 IPC sidecar(由 Tauri 启动)

```bash
python -m scribe_py ipc
```

每行读一个 JSON-RPC 请求,返回 JSON 响应或 progress 事件。

### 作为 CLI(本地调试)

```bash
python -m scribe_py transcribe path/to/audio.m4a --out ./transcripts
python -m scribe_py correct ./transcripts/audio.json --api-key sk-... --model deepseek-chat
python -m scribe_py polish ./transcripts/audio_corrected.json --api-key sk-... --model deepseek-chat
python -m scribe_py check-model
python -m scribe_py probe-audio path/to/audio.m4a
```

## 架构

```
src/scribe_py/
├── core/         # 转录抽象 (MLX / faster-whisper 双实现 + 平台路由)
├── correctors/   # LLM 字级校对 (OpenAI 兼容协议,支持 DeepSeek/OpenAI/Claude)
├── polishers/    # LLM 整篇排版
├── exporters/    # txt/srt/json/md 输出
├── ipc.py        # JSON-RPC over stdio
└── __main__.py   # 入口分发
```
