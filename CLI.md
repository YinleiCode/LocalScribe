# LocalScribe CLI — for AI coding tools

LocalScribe 提供稳定的命令行接口供 Claude Code / Hermes 等 AI 编码助手调用。

**版本**: v1.0.3

## 安装

```bash
# 一次性
ln -s "$(pwd)/bin/localscribe" /usr/local/bin/localscribe

# 验证
localscribe --help
```

## 凭据

LLM 校对/排版需要 API key:

```bash
export DEEPSEEK_API_KEY=sk-...
# 或
export OPENAI_API_KEY=sk-...
```

## 设计契约(给 AI 工具)

- **任意命令加 `--json`**:输出严格单行 JSON 到 stdout(其余日志走 stderr)
- **退出码**:0 = 成功;1 = 业务异常(如截断);2 = 输入错误;127 = 环境未就绪
- **路径解析**:相对路径基于 cwd;输出默认在 `transcripts/<audio_stem>/`

## 核心命令

### 一条龙(推荐入口)

```bash
localscribe pipeline AUDIO --json
```

转录 → LLM 校对 → 整篇排版,返回所有产物路径。

| flag | 默认 | 说明 |
|---|---|---|
| `--out DIR` | `transcripts` | 输出根目录 |
| `--language` | `zh` | Whisper 语言提示 |
| `--llm-model` | `deepseek-v4-flash` | 校对+排版用的模型 |
| `--mode` | `medium` | `light` / `medium` / `heavy` 校对强度 |
| `--concurrency` | `5` | 校对并发路数 |
| `--max-tokens` | `384000` | 排版输出上限(防截断) |
| `--no-glossary` | — | 关闭 B 两阶段术语表 |
| `--transcribe-only` | — | 只跑转录,不调 LLM |
| `--no-polish` | — | 跑转录 + 校对,不排版 |

输出 JSON:
```json
{
  "ok": true,
  "stage": "pipeline",
  "audio": "/abs/path/audio.m4a",
  "out_dir": "transcripts/audio",
  "stages": {
    "transcribe": {"files": {"txt": "...", "srt": "...", "json": "..."}, "segments": 98, "duration_s": 224, "rtf": 0.02},
    "correct": {"files": {"txt": "...", "json": "..."}, "changed": 32, "total": 98, "glossary_count": 5},
    "polish": {"file": ".../audio_完整版.txt", "char_count": 4200, "truncated": false}
  },
  "total_cost_seconds": 38.4
}
```

### 单步命令

```bash
localscribe transcribe AUDIO --json
localscribe correct  TRANSCRIPT.json --json
localscribe polish   TRANSCRIPT.json --json    # 接受 raw 或 _corrected.json
localscribe translate ARTICLE.txt --target-lang zh --json  # 翻译文章
```

**翻译命令参数**:

| flag | 默认 | 说明 |
|---|---|---|
| `--target-lang` | 必填 | 目标语言: `zh`(中文) / `en`(英文) / `ja`(日文) / `ko`(韩文) |
| `--source-lang` | 自动检测 | 源语言(可选) |
| `--llm-model` | `deepseek-v4-flash` | 翻译用的模型 |
| `--glossary` | — | 术语表 JSON 文件路径(保持专有名词一致) |

输出 JSON:
```json
{
  "ok": true,
  "stage": "translate",
  "source_file": "/abs/path/article.txt",
  "output_file": "/abs/path/article_zh.txt",
  "source_language": "ko",
  "target_language": "zh",
  "char_count": 4200,
  "truncated": false
}
```

### 工具命令

```bash
localscribe ls --json                          # 列历史库
localscribe check-model --json                 # 检查 Whisper 模型缓存
localscribe probe-audio AUDIO --json           # ffprobe 元数据
```

## 给 AI 工具的典型 prompt 范式

### 完整流程(转录 + 校对 + 排版)

```
用 LocalScribe 把这个音频转录并校对成完整文章,返回最终文件路径:

  $ localscribe pipeline /path/to/recording.m4a --json

提取 stages.polish.file 拿到完整文章路径;
若 stages.polish.truncated 为 true,需重新跑并提高 --max-tokens。
```

### 翻译文章

```
把这篇文章翻译成中文:

  $ localscribe translate /path/to/article.txt --target-lang zh --json

提取 output_file 拿到译文路径;
若 truncated 为 true,需重新跑并提高 --max-tokens。
```

### 带术语表的翻译

```
使用校对阶段提取的术语表翻译文章:

  $ localscribe translate /path/to/article.txt \
      --target-lang en \
      --glossary /path/to/glossary.json \
      --json

术语表格式: [{"term": "专有名词", "context": "上下文"}]
```

## v1.0.3 行为变化(给下游 AI 工具/解析方注意)

CLI 命令接口本身**完全向后兼容**(参数、JSON 字段、exit code 都没动),但内部行为有两处对下游解析有影响:

### 1. 多人音频的 polish 输出现在是**对话格式**

如果转录的音频含 **≥ 2 个不同 speaker**(由分人模块判定),`stages.polish.file` 的 `.txt` 内容会变成对话体:

```
**SPEAKER_A:** 第一个人说的内容...

**SPEAKER_B:** 第二个人的回应...

**SPEAKER_A:** 又轮回来...
```

每个回合以 `**NAME:**`(双星号 + 名字 + 冒号 + 双星号 + 空格)开头,回合之间空行。**NAME** 是 `SPEAKER_A` / `SPEAKER_B`(默认)或用户在 UI 改名后的真名(如`陈总`/`客户`)。

**单人音频**(或没启用分人)仍然是流文章模式,**不含任何 `**NAME:**` 标记**,完全和 v1.0.1 一样。下游想稳妥处理两种情况,可以这样判断:

```python
text = open(polish_file).read()
import re
turns = re.findall(r'^\*\*([^*\n]+?):\*\*\s*(.+?)(?=\n\*\*[^*\n]+?:\*\*|\Z)', text, re.MULTILINE | re.DOTALL)
if turns:
    # 对话体:turns = [(speaker, content), ...]
    ...
else:
    # 单人/流文章:整段当一个段落用
    ...
```

### 2. 中文输出**保证简体**

校对 / 排版 / 翻译三个阶段的 LLM prompt 都加了"必须简体中文"硬约束,并在 Python 端用 `zhconv` 把 LLM 偶尔吐出的繁体字强制转简体。下游不需要再自己跑繁简转换。

### 3. 翻译保留对话头

`localscribe translate ARTICLE.txt --target-lang en` 输入如果是对话体(含 `**NAME:**`),输出**保留所有 `**NAME:**` 标签不译**,只翻译冒号后的内容。这样英文版还是按"谁说什么"组织。

---

## 已验证场景

- 雅各书一章.m4a (3 min)→ pipeline `--transcribe-only` ≈ 5s
- 位总.m4a (96 min)→ transcribe 102s · 校对 5 路并发 ≈ 1 min · 排版 ≈ 15s
- Audio-2026-05-01-010720.m4a (3h 11min)→ transcribe 3 min · 校对 ~3 min · 排版 ~30s

## 错误处理

| exit | 含义 | 例子 |
|---|---|---|
| 0 | 成功 | 所有阶段 OK |
| 1 | 业务异常但有产出 | 排版被 max_tokens 截断 |
| 2 | 输入错误 | 音频不存在 / json 文件损坏 |
| 127 | 环境未就绪 | venv 不存在 / 模型未下载 |

stderr 始终有人类可读的进度日志,stdout 在 `--json` 时仅一行结果。
