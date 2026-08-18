# 项目开发委托书：离线录音转文字工具（OfflineScribe）

> 这是一份交给 AI 编码助手（Claude Code / Codex / Cursor 等）的完整开发说明。
> 目标：基于本地 Whisper 模型，开发一款**免费、开源、完全离线**的录音转文字桌面工具。

---

## 一、项目目标

开发一款面向普通用户的 **离线录音转文字** 桌面小工具，特点：

- ✅ **完全离线**：所有计算本地完成，录音不上传任何服务器
- ✅ **免费 + 开源**：MIT 协议发布到 GitHub
- ✅ **简单易用**：拖入录音 → 点击转录 → 得到文字，三步完成
- ✅ **支持多语言**：中文为主，兼容英文、日文等 99 种语言
- ✅ **苹果芯片优化**：在 M 系列 Mac 上用 MLX 加速；其他平台 fallback 到 faster-whisper

---

## 二、技术选型（已确定，请直接使用）

### 2.1 核心模型

- **模型名称**：Whisper large-v3-turbo
- **作者**：OpenAI
- **参数量**：809M
- **支持语言**：99 种
- **License**：MIT

### 2.2 模型来源（两个仓库，按平台选用）

| 平台 | 推荐仓库 | 推理框架 |
|---|---|---|
| Apple Silicon (M1/M2/M3/M4) | `mlx-community/whisper-large-v3-turbo` | `mlx-whisper` |
| Windows / Linux / Intel Mac | `openai/whisper-large-v3-turbo` | `faster-whisper` 或 `openai-whisper` |

**HuggingFace 链接：**
- MLX 版：https://huggingface.co/mlx-community/whisper-large-v3-turbo
- 原版：https://huggingface.co/openai/whisper-large-v3-turbo

**官方仓库与论文：**
- Whisper GitHub：https://github.com/openai/whisper
- Whisper 论文：https://arxiv.org/abs/2212.04356
- mlx-examples：https://github.com/ml-explore/mlx-examples
- faster-whisper：https://github.com/SYSTRAN/faster-whisper

### 2.3 模型本地缓存位置（开发者本机已下载）

模型已下载到本机 HuggingFace 标准缓存目录：

```
/Users/apple/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/
├── blobs/                                    # 实际权重（1.5 GB）
├── snapshots/<commit-hash>/
│   ├── README.md
│   ├── config.json
│   └── weights.safetensors                   # 主权重文件
└── refs/main
```

**开发时模型加载方式**（不要硬编码路径，用模型 ID 让框架自己解析）：

```python
# MLX 版本
import mlx_whisper
result = mlx_whisper.transcribe(
    "audio.m4a",
    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
    language="zh",
)

# faster-whisper 版本（跨平台）
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", device="auto", compute_type="auto")
segments, info = model.transcribe("audio.m4a", language="zh", beam_size=5)
```

### 2.4 依赖清单

```
# Python 3.10+
mlx-whisper>=0.4.0          # macOS Apple Silicon
faster-whisper>=1.0.3       # 跨平台 fallback
ffmpeg-python>=0.2.0        # 音频解码
huggingface_hub>=0.24       # 模型下载/缓存
```

系统依赖：
- `ffmpeg`（必须，用于音频解码）
  - macOS: `brew install ffmpeg`
  - Ubuntu: `apt install ffmpeg`
  - Windows: `choco install ffmpeg`

---

## 三、功能需求

### 3.1 必须实现（MVP）

1. **文件导入**
   - 支持拖拽：m4a / mp3 / wav / ogg / flac / aac / opus / mp4 / mov / mkv
   - 支持点击选择文件
   - 支持批量选择多个文件排队转录

2. **转录核心**
   - 自动检测语言 / 手动指定语言（默认中文）
   - 显示转录进度（百分比 + 当前段落预览）
   - 支持取消正在进行的任务

3. **结果输出**
   - 纯文本 `.txt`
   - 带时间戳字幕 `.srt`
   - JSON（含 segments + 时间戳，方便二次开发）
   - 输出文件默认保存在源文件同目录，文件名同名

4. **设置项**
   - 模型大小选择：tiny / base / small / medium / large-v3 / large-v3-turbo
   - 语言选择（含"自动检测"）
   - 输出格式勾选（txt / srt / json 多选）
   - 模型缓存目录展示与"打开目录"按钮

5. **首次启动体验**
   - 检测模型是否已下载，未下载则提示并显示下载进度
   - 检测 ffmpeg 是否安装，未安装给出安装指引

### 3.2 可选增强（v2）

- 录音机功能：直接调用麦克风录音后转录
- 说话人分离（diarization，可接 `pyannote.audio`）
- 关键词高亮 / 搜索
- 导出 Markdown 带时间戳跳转
- initial_prompt 输入框（让用户给上下文先验，提升专有名词准确率）

---

## 四、UI 与架构建议

### 4.1 推荐技术栈（开发者可选其一）

**方案 A：Tauri + React（推荐，包体积小、跨平台）**
- 前端：React + TypeScript + Tailwind
- 后端：Rust 调用 Python sidecar，或直接走 PyO3
- 包体积约 10-30 MB（不含模型）

**方案 B：Electron + Python 后端（开发快）**
- Electron 前端
- Python FastAPI 本地服务
- 包体积约 100 MB+

**方案 C：纯 Python（最快出 MVP）**
- 用 `customtkinter` 或 `flet` 写 GUI
- 单文件分发，开发周期最短
- 适合先做 v0.1 验证

**新人友好建议：先用方案 C 做 MVP 验证体验，再决定是否上 Tauri。**

### 4.2 界面草图（描述给 AI）

```
┌─────────────────────────────────────────┐
│  OfflineScribe                    ⚙ 设置 │
├─────────────────────────────────────────┤
│                                         │
│      ┌─────────────────────────┐        │
│      │   📁 拖入录音文件        │        │
│      │   或点击选择文件         │        │
│      └─────────────────────────┘        │
│                                         │
│  📋 任务队列：                            │
│  ▸ 位总.m4a       [转录中 45%]           │
│  ▸ meeting.mp3    [等待中]               │
│                                         │
│  📝 实时预览：                            │
│  ┌─────────────────────────────────┐    │
│  │ 这个我们想要做的是...           │    │
│  │ AI 驱动的工作流...             │    │
│  └─────────────────────────────────┘    │
│                                         │
│  [开始]  [暂停]  [清空]   输出: ./       │
└─────────────────────────────────────────┘
```

---

## 五、项目结构建议

```
offlinescribe/
├── README.md
├── LICENSE                    # MIT
├── pyproject.toml             # 或 package.json
├── .gitignore
├── docs/
│   ├── INSTALL.md
│   ├── USAGE.md
│   └── DEVELOPMENT.md
├── src/
│   ├── core/
│   │   ├── transcriber.py     # 转录核心，封装 mlx-whisper / faster-whisper
│   │   ├── model_manager.py   # 模型下载、缓存检查
│   │   └── audio_utils.py     # ffmpeg 探测、格式转换
│   ├── ui/
│   │   └── app.py             # GUI 入口
│   └── exporters/
│       ├── txt.py
│       ├── srt.py
│       └── json_export.py
├── tests/
│   └── test_transcriber.py
└── assets/
    └── icon.png
```

---

## 六、开源 License 与归属声明（必读）

本项目计划以 **MIT License** 发布。在 README 中**必须**包含以下归属声明：

```markdown
## Acknowledgements

This project is built upon the following open-source works:

- **Whisper** by OpenAI — MIT License
  https://github.com/openai/whisper
  Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision", 2022.
  https://arxiv.org/abs/2212.04356

- **mlx-whisper** by Apple ML Research — MIT License
  https://github.com/ml-explore/mlx-examples

- **faster-whisper** by SYSTRAN — MIT License
  https://github.com/SYSTRAN/faster-whisper

- **Whisper large-v3-turbo (MLX)** model weights from mlx-community
  https://huggingface.co/mlx-community/whisper-large-v3-turbo

All models and frameworks used are released under permissive licenses
that allow commercial and non-commercial use.
```

引用论文（BibTeX）：

```bibtex
@article{radford2022whisper,
  title={Robust Speech Recognition via Large-Scale Weak Supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2212.04356},
  year={2022}
}
```

---

## 七、给 AI 编码助手的具体任务清单

请按以下顺序实现（每完成一项跑一次冒烟测试）：

1. ☐ 初始化项目（git init + pyproject.toml + LICENSE + .gitignore）
2. ☐ 实现 `core/audio_utils.py`：ffmpeg 探测 + 时长读取 + 格式转换
3. ☐ 实现 `core/model_manager.py`：检查模型是否已缓存，未缓存则提示下载
4. ☐ 实现 `core/transcriber.py`：抽象基类 + MLX 实现 + faster-whisper 实现，按平台自动选择
5. ☐ 实现 `exporters/`：txt / srt / json 三种输出
6. ☐ 用 `argparse` 做出 CLI 版本（验证核心可跑通）：
   ```
   python -m offlinescribe transcribe input.m4a --lang zh --format txt,srt
   ```
7. ☐ 用方案 C（customtkinter / flet）做 GUI MVP
8. ☐ 写 README.md，含安装、使用、致谢、License
9. ☐ 写 tests/，至少覆盖 transcriber 和 exporters
10. ☐ 配置 GitHub Actions：lint + test + 打 release
11. ☐ （可选）打包为 .app / .exe / .AppImage

---

## 八、可参考的现有开源项目

可以借鉴 UI/UX 与代码组织（**仅参考，不要直接抄代码**）：

- **MacWhisper** （闭源商业，但 UI 是行业标杆）：https://goodsnooze.gumroad.com/l/macwhisper
- **Whisper Transcription**（开源 Mac 客户端）：https://github.com/whisper-transcription
- **WhisperKit** （Apple 平台原生）：https://github.com/argmaxinc/WhisperKit
- **buzz**（跨平台开源）：https://github.com/chidiwilliams/buzz  ← **强烈建议先看这个**
- **whisperX**（含说话人分离）：https://github.com/m-bain/whisperX

---

## 九、开发者本机环境信息（备查）

- 操作系统：macOS（Apple Silicon）
- Python：建议用 `uv` 创建独立 venv（避免污染 anaconda）
- ffmpeg：已安装于 `/opt/homebrew/bin/ffmpeg`
- 模型已缓存：`~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/`（1.5 GB）

---

## 十、验收标准

MVP 完成的标志：

- [ ] 拖入一个 1 小时的 m4a 文件，能在 5 分钟内（M 系列 Mac）完成转录
- [ ] 输出 txt/srt 文件内容正确，时间戳对齐
- [ ] 完全离线运行（拔网线测试通过）
- [ ] README 包含完整的安装步骤，新用户能在 10 分钟内跑起来
- [ ] LICENSE 文件存在，致谢部分完整
- [ ] GitHub 仓库可公开访问

---

**项目代号建议**：OfflineScribe / WhisperLocal / SilentInk / VoiceVault（任选其一或自定）

**开发周期估算**：CLI 版 1-2 天，GUI MVP 3-5 天，打包发布 +1 天。

---

> 本文档由用户与 AI 助手共同整理，可直接交给 Claude Code / Codex / Cursor 作为项目启动 brief。
> 如有疑问请回到本文档对应章节查阅，不要凭空假设技术栈或模型路径。
