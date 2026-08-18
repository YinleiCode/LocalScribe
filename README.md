# LocalScribe

> 完全离线的录音转文字桌面应用 · 可选 LLM 字级校对与整篇排版 · MIT License
> **出品方:涌智星河(SwarmPath) · 寒三修** — 隐私友好、本地可控、AI 增强的内容创作工具家族

[![Version](https://img.shields.io/badge/version-1.0.3-success)]()
[![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-blue)]()
[![Tauri](https://img.shields.io/badge/Tauri-2.10-orange)]()
[![Whisper](https://img.shields.io/badge/Whisper-large--v3--turbo-purple)]()
[![DMG](https://img.shields.io/badge/dmg-1.9%20GB-lightgrey)]()

录音文件拖进去,几分钟后得到结构化的文字稿、字幕(SRT)、整篇排版文章。
**音频不上传任何服务器**;只有在你显式启用 LLM 校对时,转录后的文字才会发送到你配置的 LLM API。

---

## 🎉 v1.0.3

**🎙️ 分人引擎升级 → Senko + CAM++(全新)**

| 部分 | 说明 |
|---|---|
| 引擎替换 | 原 resemblyzer + KMeans → **Senko + CAM++ 中文专用模型**(CoreML 加速,Apple Silicon 原生);旧 `resemblyzer_diarizer.py` 保留作为 fallback |
| 准确率 | 中文 DER 从经验值 ~20% 降到 **AISHELL-4 基准 ~13%**(CAM++ 中英混合训练,中文专项优化) |
| 速度 | **96 分钟音频 ~47 秒**(M3 实测) vs 旧版分钟级 |
| 长音频兼容 | macOS 上 senko 默认的 UMAP+HDBSCAN 在 ≥20 分钟音频会因 libomp 冲突死锁。代码层强制走 spectral clustering,长稿照常跑(单次 N² eigh 略慢但稳定) |
| 声纹库变更 | 维度从 256d(resemblyzer)→ 192d(CAM++),**用户既有的声纹样本需要重新上传**;不兼容样本会被自动跳过,日志里有提示 |
| bundled Python 集成 | `bundle-staging/python/` 内已装好 senko + 依赖(coremltools / hdbscan / umap-learn 等),装机即用 |

**💬 文章/译文 对话体渲染(全新)**

| 部分 | 说明 |
|---|---|
| 双模 polish | `article_polisher.py` 检测 segments 里 unique speakers:**≤1 人** → 现有流文章模式;**≥2 人** → 走对话 prompt,输出 `**陈总:** ... \n\n**客户:** ...` 格式。同一人连续多段自动合并为一个回合 |
| 长稿对话分块 | 分块时按 **speaker 边界**优先切,避免把一个人发言切碎在两个块里 |
| 译文保留对话头 | `article_translator.py` 检测到对话格式时,prompt 命令 LLM "保留所有 `**NAME:**` 标签,只翻译冒号后内容"。英文目标语言下,"陈总" 这种人名也不译,保持回合归属 |
| UI 自动渲染 | 文章 / 译文 tab 自动识别 `**NAME:**` 头,渲染成和原文 tab 同色 SpeakerChip + 内容,视觉一致 |
| 改名全链路同步 | 在原文点 chip 改名("SPEAKER_A → 陈总"),**自动同步到**:raw segments + corrected segments + polished 文章正文 + translated 译文正文,并持久化到对应 JSON。即使文章在改名 *之前* 就生成过,**切到文章 tab 时也会按"首次出现顺序"自动对齐替换** |
| inline 改名输入框 | 替换 `window.prompt()`(Tauri WKWebView 里被禁用),用 inline 输入框,默认 14 字符宽,Enter 提交、Esc 取消 |

**🐛 Bug 修复**

| Bug | 修复 |
|---|---|
| UI 版本号显示为旧版本 | `src/App.tsx` 和 `src/components/SettingsDialog.tsx` 里硬编码的版本号没跟版本同步(本次也一起跟到 `v1.0.3`)。后续 bump 版本时记得同步这两处,或者改成从 `package.json` / `Cargo.toml` 动态读取 |
| 取消任务后新任务卡"等待" | `usePipeline.ts` 的 `runningRef` 用 `useRef` 在 hook 内部,而 `cancelTask` 是模块级独立函数,改不到 hook 内部的 ref。导致点取消后 `runningRef` 永远是 `true`,新拖入的音频永远停在"等待"。把状态提到模块级 `pipelineState`(`running` + `cancelledIds`),`cancelTask` 立即把 `running` 复位、把任务 id 塞进 `cancelledIds`,流水线 async 块每次 await 完检查 `cancelledIds` 决定要不要写回结果。代价:被取消任务的 Python 工作仍会跑完(单进程 sidecar 没有中途打断机制),但 UI 和后续任务都不再被阻塞 |
| 转录进度反复跳跃(25%→16%→30%→20%) | 同一份 `progress` 状态被两个 writer 用不同单位写:真进度来自 Python VAD 分块(`{current: 3, total: 12}` = 25%),伪进度是前端基于音频时长估算的百分比(`{current: 30, total: 100}` = 30%)。两者把 `total` 字段当不同含义,UI 在两套单位间切换时百分比剧烈跳变;当伪进度因为音频时长 probe 偏差而严重超估时,真进度一来还会暴跌。修法:**统一刻度** —— 真进度写入前先归一化成 `{current: pct, total: 100}`,然后**真假两路都加"进度不许倒退"闸**(新百分比 < 旧的就拒绝)。UI 现在严格单调递增,fake 估错也只是平台期不会回退 |
| Polish 长稿被截断(原 25081 字仅生成 10210 字 41%) | DeepSeek 等 LLM 有"单次输出 token 数"硬上限(deepseek 8K tokens,即使客户端 `max_tokens=384000`),长稿被服务端截断。`article_polisher.py` 加分块逻辑:按 segment 边界贪心切到 ≤ 4000 字/块,每块独立 polish 再拼接。中文 4000 字 → ~5K tokens 输出,稳进 deepseek 8K 上限内,任意长度都能完整排版 |
| LLM 偶尔输出繁体字 | DeepSeek 在长上下文里偶尔混出繁体段落。两层防御:**(1) prompt** —— 校对 / 排版 / 翻译的中文 prompt 全部加"必须输出简体中文(GB18030 字符集),严禁繁体字"硬约束;**(2) 代码** —— 用 `zhconv` 库在 LLM 输出后强制把繁体字转简体,纯字典查表(~100KB,无 C 依赖),已装进 bundled Python |
| 下载文件后找不到 | 两个问题叠加。**(a)** `save()` 的 `defaultPath` 只给了文件名没给目录,save 对话框开在"上次保存位置"用户找不到 → 改成默认到 `~/Downloads/`,并在保存成功后弹对话框告知实际路径。**(b)** `tauri.conf.json` capabilities 里 `fs:default` 不允许写入用户选定的任意路径,writeTextFile 静默失败 → 加 `fs:allow-write-text-file` / `fs:allow-write-file` 等显式权限,并通过 `fs:scope` 限定到 `$HOME/**` / `$DOWNLOAD/**` 等安全范围 |
| 下载文件名带 `.*` 后缀(`xxx.txt.*`) | save 对话框的 filters 配置 `extensions: ["*"]` 让 macOS 字面把 `.*` 追加成扩展名。改成从输入文件名提取真实扩展(`.txt` / `.md` / `.srt` / `.json`),filter 名也对应("Text" / "Markdown" 等) |

## 🎉 v1.0.2

**🌐 文章翻译(全新功能)**

| 部分 | 说明 |
|---|---|
| 翻译引擎 | 基于排版后的完整文章,支持翻译到中文/英文/日文/韩文 |
| 独立配置 | 设置 → 校对 → 翻译模型(独立于校对和排版的模型配置) |
| 高级参数 | 可单独调整翻译的温度、最大输出等参数 |
| 术语一致性 | 自动使用校对阶段提取的术语表,保持专有名词翻译一致 |
| 使用方式 | 文章 Tab → 点击底部"翻译"按钮 → 选择目标语言 → 译文 Tab 查看结果 |
| 导出支持 | 译文可导出为 .txt / .md 格式 |

**⚙️ 配置优化**

| 改进 | 说明 |
|---|---|
| 翻译配置独立 | 新增 `TranslationSettings` 结构,翻译模型和参数独立配置 |
| 设置结构清晰 | 校对、排版、翻译三个功能的配置层级更清晰 |
| 向后兼容 | 旧版本设置文件自动迁移,添加默认翻译配置 |

**🐛 Bug 修复**

| Bug | 修复 |
|---|---|
| 翻译按钮禁用 | 修复 Rust 后端缺少 `translation` 字段导致翻译功能无法使用的问题 |
| 设置加载失败 | 修复前端加载设置时因缺少 `translation` 配置导致的错误 |
| 点击翻译目标语言无反应 | Tauri v2 默认期望 camelCase 参数,但前端发的是 snake_case (`target_language` 等),导致 `translate_article` 命令报 `missing required key targetLanguage`。给 `correct_segments` / `polish_article` / `translate_article` 加 `#[tauri::command(rename_all = "snake_case")]`,前后端参数命名对齐 |
| LLM 高级参数被忽略(隐藏 bug)| 同一原因下,`polish` / `correct` 的 `base_url`、`max_tokens`、`temperature` 等参数因为是 `Option<T>`,Tauri 静默丢弃未匹配的 key,Rust 一直在用代码里的默认值 —— 用户在设置面板里调的 LLM 参数实际从未生效。本次修复后这些参数才真正传到后端 |

## 🎉 v1.0.1

(以 git commit `07587f9` v1.0.0 为基线)

**👥 说话人分离(全新功能)**

| 部分 | 说明 |
|---|---|
| 引擎 | 新增 `scribe-py/diarizers/` 模块。silero-vad 找说话区间 + 拼接 + 保留时间映射 → Resemblyzer 提声纹 → KMeans 聚类。自动 K 检测(silhouette 扫描 K=2-8 选最优,单人时 silhouette < 0.10 判 1 簇) |
| 声纹库 | 设置 → 说话人 → 上传声纹样本(每人 5-30 秒纯人声),系统提 256 维 embedding 存进 settings.json · 后续录音 cosine 相似度 ≥ 0.65 自动用真实姓名替代 SPEAKER_A/B |
| 段落显示 | 每段前彩色 chip(8 路调色板循环)· 默认对话视图(按说话人合并连续段成 turn,< 1.2 秒间隔合并)· 标题栏切换 [对话 \| 时间戳] |
| 点击 chip 改名 | 全局生效 — 当前任务所有标着 `SPEAKER_A` 的段一起改成"三修",同步写回 raw / corrected JSON,重启不丢 |
| 重新跑分人按钮 | 原文 tab 顶部:不重新转录,30-60 秒重新跑 diarization(老任务用新算法刷一遍) |
| 导出带说话人 | txt / srt 段前加 `[说话人]` 前缀;md 按说话人切 H2 章节 |
| Pipeline 集成 | 设置开启后,转录完自动接力跑 diarize · 任务卡多一个"分人"阶段进度 |

**🐛 Bug 修复**

| Bug | 修复 |
|---|---|
| LLM 校对丢失 speaker 字段 | `_correct_batch` 创建新 Segment 时漏了 speaker,校对后段落不再有标签。补上字段透传(同时修 Pass 1 取消 / Pass 2 失败兜底 / 缺批兜底 三条路径) |
| .app 打包反复装空壳 | Tauri DMG bundle 步骤被旧挂载点阻塞时,build-app.sh 会继续装 14 MB 空壳到 /Applications/。加铁腕清理(扫所有 LocalScribe 挂载点 detach,前后两次)+ 注入后大小硬校验(< 1 GB 直接 abort)+ 容忍 Tauri DMG 步骤失败(我们自己重打 DMG) |

**⚙ 任务队列优化**

| 改进 | 说明 |
|---|---|
| 新文件加进来自动选中 | `tasks.add()` 总是把新任务设为 active(之前只在 active 为空时设),拖入即看到正在处理的项 |
| 跑动中任务置顶 | 新增 stagePriority 排序:跑动 > 等待/暂停 > 完成 > 失败/取消 |
| 跑动中视觉强化 | 左侧 3px 蓝色 accent 条 + 浅蓝背景 + 右上角脉动小蓝点 |
| 完成项从队列移走 | transcribed / corrected / polished 三个终态从队列过滤掉,只留在「历史库」显示 |
| 队列徽章数字 | 只数"在跑 + 等待 + 失败",不算已完成 |

## 🎉 v1.0.0 · 首个正式版

| 类别 | 改进 |
|---|---|
| 🚀 **可分发** | 自包含 .dmg(~1.9 GB · 内置 Python 3.12 + 模型 + ffmpeg)— 双击装到 Applications,**用户什么都不用装** |
| 🎯 **不丢段** | VAD 引导转录:silero-vad 先切说话区间再逐段送 Whisper,解决长 chunk 漏段 bug |
| ⚡ **更快校对** | 默认并发 5 → **15**,批大小 20 → **30**,综合 4-5x 加速;**急速模式**再快 30% |
| 📁 **数据规范** | 用户数据搬到 `~/Library/Application Support/LocalScribe/`,卸载/升级不丢 |
| 🛡️ **首启引导** | 模型缺失时显示三步引导页(下载 → 放指定目录 → 重新检测) |
| 🏗️ **构建系统** | 新 `build-bundle.sh` + `build-app.sh`,一行命令出可分发 .dmg |

老用户升级:settings 自动迁移到新默认值 — 启动时检测旧 5/20 默认 → 自动改 15/30 并写回。

---

## ✨ 特性

- **快**:中文默认使用非自回归 SenseVoice;不可用时自动回退到 Apple Silicon MLX Whisper 或跨平台 faster-whisper
- **不丢段**:VAD 引导转录,先切说话区间再逐段识别,避开长音频整窗漏段
- **准**:SenseVoice 中文优先 + 音频自适应预处理 + ASR 质量报告 + 热词/术语一致性检查
- **离线**:转录环节零网络。LLM 校对可选,默认关闭
- **省**:DeepSeek-v4-flash 校对 1 小时音频 ~0.5 元
- **专业**:VSCode 风格界面 · **15 路并发校对**(默认)· 急速模式开关 · 暂停/继续/取消 · 支持 384K token 输出
- **历史库**:自动持久化所有转录到 `transcripts/<文件名>/`,以后随时载入查看
- **CLI 友好**:全部功能可通过命令行 + JSON 协议给 AI 编码工具(Claude Code / Hermes)调用
- **开箱即用**:提供自包含 `.dmg`(~1.9 GB · 内置 Python + 模型 + ffmpeg),双击装到 Applications 即用

---

## ASR 准确率实测（2026-07-14）

LocalScribe 使用两轮真实中文录音人工核对来验证默认转录链路。第二轮覆盖电话业务、节目、讲道/查经共 10 份录音，固定抽取 50 段、约 8.3 分钟音频，人工逐段确认 2244 个参考字符；评分时忽略标点和空白。

| 模型 | 字符错误率 CER | 字符准确率 | 完全正确段 | 产品定位 |
|---|---:|---:|---:|---|
| **SenseVoice（默认）** | **0.98%** | **99.02%** | **43/50** | 默认中文主模型 |
| Paraformer | 6.06% | 93.94% | 15/50 | 仅提供疑点证据 |
| Qwen3-ASR 1.7B | 6.64% | 93.36% | 15/50 | 仅提供疑点证据 |

分场景 SenseVoice 字符准确率：电话业务 98.87%、节目 98.81%、讲道/查经 100.00%。第一轮独立的 18 段严格人工真值评分为 98.22%，并完成了重复推理稳定性修复与 5/5 一致性验证。

基于人工核对结果，当前产品策略为：

1. SenseVoice 保持默认主模型；标准模式不额外加载辅助模型。
2. Paraformer 和 Qwen3-ASR 只用于定位疑点，不自动覆盖主转录。
3. `strong` 本地高质量模式保持手动可选且默认关闭。
4. 不针对单条录音写死替换规则；行业术语、人名和缩写优先使用热词与人工词表。

这组结果证明当前版本已满足本项目普通中文录音的内部可用性目标，但它不是公开盲测排行榜成绩：样本规模仍较小，尚未充分覆盖远场麦克风、强噪声、多人重叠、方言、实时流式和跨语言场景。因此不能把 99.02% 直接等同于所有环境下的行业通用准确率。

完整过程和逐条评分见 `output/asr_direct_gold_round2_v2_20260714/第二轮通用ASR结论.md`。

---

## 📥 安装

### 路线 A · 直接装 .dmg(推荐普通用户)

如果作者/朋友给了你 `LocalScribe_1.0.3_aarch64.dmg`(~1.9 GB):

```
1. 双击 LocalScribe_1.0.3_aarch64.dmg
2. 拖 LocalScribe 图标到 Applications 文件夹
3. 启动台 / Finder 找到 LocalScribe → 右键打开(首次会问"未验证开发者")
4. 直接用 — Python / Whisper 模型 / ffmpeg 全部内置,**不用装任何东西**
```

DMG 包含:
- **可重定位 Python 3.12** + mlx-whisper / silero-vad / openai 等所有依赖
- **Whisper large-v3-turbo** 权重(1.5 GB)
- **ffmpeg / ffprobe** 静态二进制(arm64)

用户数据(转录、文章库、设置)自动放到 `~/Library/Application Support/LocalScribe/`,卸载/升级不丢。

### 路线 B · 源码自构建(开发者)

```bash
git clone <仓库地址> LocalScribe
cd LocalScribe
./install.sh           # 自动:装 ffmpeg/uv/pnpm/Rust → 装 Python 依赖 → 下模型 → 构建 dev .app
```

**国内网络加速**:
```bash
HF_MIRROR=1 ./install.sh    # 用 hf-mirror.com + 清华源 + npmmirror
```

**仅装依赖,不构建 .app**:
```bash
SKIP_BUILD=1 ./install.sh   # 用源码 dev 模式跑:pnpm tauri dev
```

dev .app 出在 `src-tauri/target/release/bundle/macos/LocalScribe.app`(依赖项目源码,不便分发)。

### 路线 C · 自己出可分发 .dmg

```bash
./install.sh                       # 先把 .venv + 模型准备好
./build-app.sh                     # 自动:下 python-build-standalone + ffmpeg → 注入 .app → 出 .dmg
# 产物: src-tauri/target/release/bundle/dmg/LocalScribe_1.0.3_aarch64.dmg (~1.9 GB)
# (DMG 文件名版本号由 build-app.sh 自动从 tauri.conf.json 读取,无需手动改脚本)
```

`build-app.sh` 做的事:
1. `build-bundle.sh` 准备 staging:可重定位 Python + 装依赖 + ffmpeg + 模型
2. `pnpm tauri build` 出基础 .app
3. `tar` 管道把 staging 注入 `.app/Contents/Resources/`(剥离 `com.apple.provenance` xattr,绕开 macOS Tahoe 的 .app 写保护)
4. `hdiutil` 重新生成 UDZO 压缩 .dmg

### 配置 DeepSeek API Key(可选,启用 LLM 校对/排版)

1. 申请 Key:https://platform.deepseek.com(注册送额度,够测试)
2. 启动 LocalScribe → ⚙ 设置 → 校对
3. 启用 LLM 校对 → 弹隐私确认 → 粘贴 Key → 保存

**Key 存放在 macOS 系统钥匙串,不会上传任何地方。**

### 首次启动 macOS 提示"无法验证开发者"

我们的 `.app` 未做苹果开发者签名。绕过:
```bash
xattr -cr /Applications/LocalScribe.app
```
或:系统设置 → 隐私与安全性 → 滚到底 → 点"仍要打开"。

### Intel Mac / Linux / Windows

当前 `install.sh` 只针对 Apple Silicon 调试过。其他平台:
- 模型路径切到 `deepdml/faster-whisper-large-v3-turbo-ct2`(CT2 格式)
- 后端 `--backend=ct2`(faster-whisper)
- 速度慢约 5-10 倍(无 GPU)

跨平台分发版仍在 roadmap,见下文。

---

## 🎯 使用

### 在 GUI 里

1. 拖入音频/视频文件到左侧 DropZone(支持 m4a / mp3 / wav / mp4 / mov 等)
2. 自动转录,任务卡显示进度
3. 切到 **校对** tab → 点 **开始校对** 或 **校对+排版(一键)**
4. 切到 **文章** tab 看完整排版稿
5. 任意 tab 都能 **复制 / 导出 .txt / .srt / .md / .json**

### CLI(供 AI 工具调用)

```bash
# 一句话搞定整条流水线
./bin/localscribe pipeline /path/to/audio.m4a --json

# 仅转录
./bin/localscribe transcribe audio.m4a --json

# 列出历史库
./bin/localscribe ls --json
```

详见 [CLI.md](./CLI.md)。

### 链到 PATH

```bash
ln -s "$(pwd)/bin/localscribe" /usr/local/bin/localscribe
localscribe --help
```

---

## 🛠 从源码构建

适合开发者 / 想自己改的人。当前只构建 macOS Apple Silicon。

### 依赖

| 工具 | 用途 | 版本 |
|---|---|---|
| Rust + Cargo | Tauri 后端 | 1.77+ |
| Node + pnpm | React 前端 | Node 20+, pnpm 9+ |
| uv | Python 依赖管理 | 0.4+ |
| Python | 转录 sidecar | 3.10+ |
| ffmpeg | 音频解码 | 任意现代版本 |

### 安装步骤

```bash
git clone <repo-url> LocalScribe
cd LocalScribe

# 1. 前端依赖
pnpm install

# 2. Python venv + sidecar
uv venv
uv pip install --python .venv/bin/python -e scribe-py
uv pip install --python .venv/bin/python silero-vad

# 3. 模型(1.5 GB · Apple Silicon 首选 MLX 版)
#    ⚠️ 默认下到项目内 ./models/whisper-large-v3-turbo/,这样整个 LocalScribe 文件夹可以直接拷给别人
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mlx-community/whisper-large-v3-turbo',
                  local_dir='./models/whisper-large-v3-turbo',
                  local_dir_use_symlinks=False)
"
# 或者把别人发给你的 weights.safetensors / config.json 直接放到:
#   LocalScribe/models/whisper-large-v3-turbo/
# 也可通过环境变量指向其他位置:
#   export LOCALSCRIBE_MODEL_DIR=/path/to/whisper-large-v3-turbo

# 4. 配置 API key(可选,用于 LLM 校对)
echo '{"keys": {"deepseek": "sk-..."}}' > .dev-secrets.json
chmod 600 .dev-secrets.json

# 5. 开发模式(热重载)
pnpm tauri dev

# 5b. 网页版开发模式(两个终端)
pnpm web:api   # 本机 API: http://127.0.0.1:8765
pnpm web:ui    # 网页 UI: http://127.0.0.1:11517

# 6. 生产构建
pnpm tauri build
# 产物:src-tauri/target/release/bundle/macos/LocalScribe.app
#       src-tauri/target/release/bundle/dmg/LocalScribe_1.0.3_aarch64.dmg
```

### 注意事项

- `pnpm tauri build` 出的是 **dev 版 .app** — 依赖 `<repo>/.venv/bin/python3` 绝对路径,只能你这台机器运行
- 网页版复用同一套 React UI 和 Python 转录/校对/纪要逻辑。浏览器无法直接读取本机文件路径,所以音视频会先上传到本机 API 的 `uploads/` 目录,再进入原流水线;数据仍在本机。
- 想给别人用:跑 `./build-app.sh` 出**自包含 .dmg**(~1.9 GB,内置 Python + 模型 + ffmpeg)

---

## 📁 项目结构

```
LocalScribe/
├── README.md                    本文档
├── CLI.md                       AI 工具调用 CLI 接口
├── PROJECT_BRIEF.md             项目需求文档
├── install.sh                   开发环境一键安装 · 装依赖 + 下模型 + dev build
├── build-bundle.sh              准备 staging:可重定位 Python + 装依赖 + ffmpeg + 模型
├── build-app.sh                 出自包含 .dmg(staging 注入 .app + 重打 dmg)
├── package.json                 前端依赖
├── tailwind.config.cjs          VSCode dark+ 配色
├── index.html                   Vite 入口
│
├── bin/
│   └── localscribe              shell wrapper(可链到 PATH)
│
├── src/                         React 前端(TypeScript + Tailwind)
│   ├── App.tsx                  主页面 · TitleBar / Sidebar / StatusBar
│   ├── components/              组件(VSCode 风格)
│   ├── stores/                  Zustand 状态管理
│   ├── hooks/                   usePipeline / 暂停取消
│   └── lib/ipc.ts               Tauri 命令类型化封装
│
├── src-tauri/                   Tauri 2 后端(Rust)
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   └── src/
│       ├── main.rs / lib.rs     Tauri 入口
│       ├── commands.rs          18 个 #[tauri::command]
│       ├── sidecar.rs           Python sidecar 进程管理 + IPC
│       ├── library.rs           transcripts/ 持久化
│       └── secrets.rs           keychain 封装
│
├── scribe-py/                   Python sidecar(转录核心)
│   ├── pyproject.toml
│   └── src/scribe_py/
│       ├── __main__.py          CLI 入口(8 个子命令)
│       ├── ipc.py               JSON-RPC over stdio
│       ├── core/
│       │   ├── transcriber_mlx.py    MLX 实现 + 4 层幻觉防御
│       │   ├── transcriber_ct2.py    faster-whisper 跨平台
│       │   └── audio.py              ffprobe
│       ├── correctors/
│       │   ├── openai_compatible.py  并发校对 + 术语表
│       │   └── prompts.py            light/medium/heavy 三档
│       └── polishers/
│           └── article_polisher.py   整篇排版
│
├── models/                     Whisper 权重(gitignored,1.5 GB · install.sh 自动下载)
│   └── whisper-large-v3-turbo/
│       ├── weights.safetensors
│       └── config.json
│
├── src-tauri/bundle-staging/   .dmg 打包暂存区(gitignored,~3 GB)
│   ├── python/                 python-build-standalone + 装好的 site-packages
│   ├── scribe-py/              我们的 Python 包(打包模式用 site-packages 副本)
│   ├── models/                 模型副本
│   └── bin/ffmpeg + ffprobe    静态二进制
│
└── transcripts/                自动保存的转录结果(每个音频一个子目录)
    ├── <stem1>/
    │   ├── <stem1>.txt / .srt / .json     转录原文
    │   ├── <stem1>_corrected.txt / ...     LLM 校对后
    │   ├── <stem1>_diff.txt                修改对比
    │   └── <stem1>_完整版.txt              整篇排版稿
    └── <stem2>-20260501-1610/              旧版本归档

# 装到 .app 后,用户数据搬到这里(macOS 标准位置):
~/Library/Application Support/LocalScribe/
    ├── transcripts/
    └── articles/
```

---

## 🔒 隐私模型

| 数据 | 是否离开本机 |
|---|---|
| 音频文件 | **永不上传**(转录全程本地 GPU) |
| 转录文字(关 LLM 时) | 只在本机 |
| 转录文字(开 LLM 时) | 发送到你配置的 LLM 提供商进行校对/排版 |
| API Key | 存 macOS 钥匙串,从不离开本机 |
| 历史库 | `transcripts/` 文件夹,纯本地 |

启用 LLM 校对时会弹出隐私提示,需用户明确确认。

---

## 🧠 技术亮点

### 转录幻觉防御(4 层架构)

Whisper 在静音段会"幻觉"出训练集高频短语(感谢观看 / 请订阅 / Fro Fro)。我们做了:

1. **VAD 输入清理**:silero-vad 检测说话区间,非语音段直接丢弃
2. **解码硬化**:`condition_on_previous_text=False` 切自反馈循环 + 收紧 `no_speech` / `compression_ratio` / `logprob` 阈值
3. **置信度自校**:模型自报的 `avg_logprob < -1.0` 段直接丢弃
4. **统计后处理**:重复检测 / 字符密度异常 / 段间相似度 / 已知幻觉短语黑名单

详见 `scribe-py/src/scribe_py/core/transcriber_mlx.py`。

### LLM 校对优化

- **B 两阶段**:Pass 1 扫全文提取专有名词术语表 → Pass 2 每批校对带词表保持跨段一致性
- **15 路并发**(默认 · 可调):`ThreadPoolExecutor` + DeepSeek API,3 小时音频 ~6 分钟 → ~1.5 分钟
- **急速模式**:跳过 Pass 1 术语提取,通用内容再快约 30%(设置 → 校对 → 急速模式)
- **暂停/继续/取消**:reader 线程 + worker 池架构,校对中也能实时响应控制命令
- **失败隔离**:某批失败保留原文,不让一个错误拖垮整篇

### VAD 引导转录(解决 Whisper 漏段)

Whisper 处理 30 秒以上连续片段时,内部 chunk 决策有时会**整窗丢段** — 同一段音频
单独切出来送给 Whisper 能识别,放在长音频里又会被跳过。修复方式:

1. `silero-vad` 先扫整段音频 → 输出说话区间时间戳
2. 合并间隔 < 0.6 秒的相邻区间(避免短片段上下文不足)
3. 拆开 > 25 秒的(避开 Whisper chunk 边界)
4. 每个区间单独 ffmpeg 切片 + mlx-whisper 转录,最后按全局时间拼接

实测:之前会丢的"经文 1-3 节"现在完整出现。代价 RTF 约 0.04 → 0.06(仍远低于实时)。
默认开启,环境变量 `LOCALSCRIBE_VAD_GUIDED=0` 可关。

### 截断检测

LLM 输出有 token 限制,超长内容会被截断。我们:
- 默认 `max_tokens=384000`(DeepSeek 上限)
- 检测 `finish_reason=="length"`,在文章页显示醒目警告
- 引导用户提高 max_tokens 重跑

---

## 🚧 路线图

### 当前开发优先级

1. **P0 · 转文字** — 先完成准确率、漏段、长音频稳定性、时间戳和导出结果的完整验收。
2. **P1 · 人声/说话人分离 + 时间轴光标** — 转录验收通过后,完成说话人边界、播放光标和文本高亮联动。
当前暂停实时录音、系统音频、Watch Folder、YouTube 导入和多平台扩展等非核心方向,避免分散转文字主链路的开发与验证。

- [x] MLX + faster-whisper 双后端
- [x] LLM 校对 + 排版
- [x] 历史库 + 重复检测
- [x] 4 层幻觉防御
- [x] CLI + JSON 协议
- [x] **15 路并发**校对 + 暂停/取消 + 急速模式
- [x] **VAD 引导转录** — 解决 Whisper 长 chunk 漏段
- [x] **SenseVoice 中文优先** — 缓存可用时由 auto 后端默认选择,Whisper 作为跨平台回退
- [x] **ASR 人工真值回归** — 两轮真实录音核对,第二轮 50 段字符准确率 99.02%
- [x] **可分发 .dmg**(~1.9 GB · 内置 Python + 模型 + ffmpeg)— 双击装到 Applications 即用
- [x] **模型缺失引导页** — 启动时若没找到权重,UI 引导用户放入正确目录
- [x] **说话人分离** — Senko + CAM++ 中文优先,Resemblyzer fallback,支持声纹库与改名同步
- [ ] **代码签名 + 公证**(Apple Dev ID · 消除"未验证开发者"提示)
- [ ] **首启 wizard**(语言 / 模型大小 / 镜像三步引导)
- [ ] **Windows / Linux 构建**
- [ ] **Live recording**(直接调系统麦克风)

---

## 🌌 关于涌智星河 / SwarmPath

LocalScribe 由 **涌智星河(SwarmPath) · 寒三修** 出品,是其旗下的开源产品之一。
涌智星河致力于打造一系列**隐私友好、本地可控、AI 增强**的工具,帮助个人与小团队
完成从录音 → 文字 → 知识 → 决策的完整闭环。

| 产品 | 定位 |
|---|---|
| **LocalScribe**(本仓库) | 离线录音转文字 · 可选 LLM 校对 · 文章库 · CLI 友好 |
| 其他兄弟项目 | 持续构建中 — 关注 SwarmPath 后续发布 |

所有代码以 **MIT 协议**开源,商业与非商业使用皆免费。问题反馈 / 贡献欢迎提 Issue / PR。

---

## 🙏 致谢

LocalScribe 站在以下开源项目肩膀上:

- **[Whisper](https://github.com/openai/whisper)** © OpenAI · MIT License
  Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision", 2022
  https://arxiv.org/abs/2212.04356
- **[mlx-whisper](https://github.com/ml-explore/mlx-examples)** © Apple ML Research · MIT License
- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** © SYSTRAN · MIT License
- **[silero-vad](https://github.com/snakers4/silero-vad)** © Silero Team · MIT License
- **[Tauri](https://tauri.app)** · **[React](https://react.dev)** · **[DeepSeek API](https://api-docs.deepseek.com)**

模型权重:[`mlx-community/whisper-large-v3-turbo`](https://huggingface.co/mlx-community/whisper-large-v3-turbo)

---

## 📜 License

MIT License — 见 [LICENSE](./LICENSE)

```
@article{radford2022whisper,
  title={Robust Speech Recognition via Large-Scale Weak Supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2212.04356},
  year={2022}
}
```
