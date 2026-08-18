# LocalScribe 客户交付说明

## 交付内容

- 完整项目源码：公开仓库 `main` 分支。
- macOS 离线 App：GitHub Releases 中的分卷压缩包。
- SHA-256 校验文件：用于确认所有分卷下载完整。

当前交付版保留本地录音转写、简体中文与标点规范化、说话人分离、音频同步光标、按说话人归类及分段播放。会议纪要和 Hermes 功能不在本版本中。

## 运行要求

- Apple Silicon Mac（M1/M2/M3/M4 或更新机型）。
- macOS 13 或更高版本。
- 至少 12 GB 可用磁盘空间，用于下载、合并和解压离线 App。

## 获取源码

```bash
git clone https://github.com/YinleiCode/LocalScribe.git
cd LocalScribe
```

项目不会从 Git 仓库下载或提交录音、转录结果、模型缓存、密钥和本地运行数据。

## 安装 App

1. 在仓库的 Releases 页面下载同一版本的全部 `part-*` 文件及 `SHA256SUMS.txt`。
2. 在下载目录执行：

```bash
shasum -a 256 -c SHA256SUMS.txt
cat LocalScribe_1.0.3_macOS_AppleSilicon.tar.gz.part-* > LocalScribe_1.0.3_macOS_AppleSilicon.tar.gz
tar -xzf LocalScribe_1.0.3_macOS_AppleSilicon.tar.gz
```

3. 将 `LocalScribe.app` 移到“应用程序”文件夹。
4. 首次启动时在 Finder 中右键点击 App，选择“打开”。

App 已包含离线转写和说话人处理所需的本地运行环境及模型，正常使用不依赖外部大模型 API。

## 源码验证

```bash
pnpm install
pnpm build
pnpm test:article-speaker-groups
pnpm test:transcript-sync
```

完整离线 App 的构建还需要准备项目文档中说明的本地 Python 运行环境和模型资源。
