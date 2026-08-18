//! Articles 知识库 — 把生成的"完整版"排版稿固化为稳定路径的 markdown 文档,
//! 带 YAML frontmatter,供 AI agent 通过 glob `articles/*.md` 读取。

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const DIR_NAME: &str = "articles";

pub fn articles_root() -> PathBuf {
    crate::library::project_root().join(DIR_NAME)
}

fn ensure_dir(p: &Path) -> Result<()> {
    std::fs::create_dir_all(p).with_context(|| format!("create {}", p.display()))?;
    Ok(())
}

fn now_iso8601() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs / 86400 + 719162;
    let (year, month, day) = days_to_ymd(days);
    let hour = (secs % 86400) / 3600;
    let minute = (secs % 3600) / 60;
    let sec = secs % 60;
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hour, minute, sec
    )
}

fn days_to_ymd(days: i64) -> (i64, i64, i64) {
    let mut d = days;
    let mut year = 0i64;
    loop {
        let y_days = if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
            366
        } else {
            365
        };
        if d < y_days {
            break;
        }
        d -= y_days;
        year += 1;
    }
    let month_days = if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 0;
    while d >= month_days[month] {
        d -= month_days[month];
        month += 1;
    }
    (year, (month + 1) as i64, (d + 1) as i64)
}

/// Sanitize a user-provided title into a safe filename.
///
/// 允许中文、字母、数字、下划线、连字符、空格;其他字符替换为 `_`。
fn safe_filename(title: &str) -> String {
    let trimmed = title.trim();
    if trimmed.is_empty() {
        return "untitled".into();
    }
    let mut out = String::with_capacity(trimmed.len());
    for ch in trimmed.chars() {
        let safe = ch.is_alphanumeric() || ch == '-' || ch == '_' || ch == ' ' || (ch as u32) > 127; // CJK and other unicode kept
        out.push(if safe { ch } else { '_' });
    }
    // Collapse runs of spaces / underscores
    let cleaned: String = out.split_whitespace().collect::<Vec<_>>().join(" ");
    cleaned.chars().take(80).collect()
}

#[derive(Debug, Deserialize)]
pub struct SaveArticleArgs {
    pub title: String,
    pub content: String,
    /// 元数据字段会写入 frontmatter
    pub source_audio: Option<String>,
    pub source_stem: Option<String>,
    pub duration_seconds: Option<f64>,
    pub model: Option<String>,
    pub based_on: Option<String>, // "corrected" or "raw"
    pub tags: Option<Vec<String>>,
    pub note: Option<String>,
    /// false (default) = 若同名存在则在末尾加序号;true = 直接覆盖
    #[serde(default)]
    pub overwrite: bool,
}

#[derive(Debug, Serialize)]
pub struct ArticleMeta {
    pub title: String,
    pub filename: String,
    pub path: String,
    pub source_audio: Option<String>,
    pub source_stem: Option<String>,
    pub duration_seconds: Option<f64>,
    pub char_count: usize,
    pub model: Option<String>,
    pub based_on: Option<String>,
    pub tags: Vec<String>,
    pub note: Option<String>,
    pub created_at: String,
    pub modified_at: String,
}

fn unique_path(base: &Path, name: &str) -> PathBuf {
    let primary = base.join(format!("{name}.md"));
    if !primary.exists() {
        return primary;
    }
    for n in 2..1000 {
        let candidate = base.join(format!("{name} ({n}).md"));
        if !candidate.exists() {
            return candidate;
        }
    }
    primary
}

fn render_frontmatter_and_body(
    args: &SaveArticleArgs,
    char_count: usize,
    created_at: &str,
) -> String {
    let mut s = String::new();
    s.push_str("---\n");
    s.push_str(&format!("title: {}\n", yaml_escape(&args.title)));
    if let Some(v) = &args.source_audio {
        s.push_str(&format!("source_audio: {}\n", yaml_escape(v)));
    }
    if let Some(v) = &args.source_stem {
        s.push_str(&format!("source_stem: {}\n", yaml_escape(v)));
    }
    if let Some(v) = args.duration_seconds {
        s.push_str(&format!("duration_seconds: {}\n", v));
    }
    s.push_str(&format!("char_count: {char_count}\n"));
    if let Some(v) = &args.model {
        s.push_str(&format!("model: {}\n", yaml_escape(v)));
    }
    if let Some(v) = &args.based_on {
        s.push_str(&format!("based_on: {}\n", yaml_escape(v)));
    }
    if let Some(tags) = &args.tags {
        if !tags.is_empty() {
            s.push_str("tags: [");
            for (i, t) in tags.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                s.push_str(&yaml_escape(t));
            }
            s.push_str("]\n");
        }
    }
    s.push_str(&format!("created_at: {created_at}\n"));
    if let Some(v) = &args.note {
        if !v.trim().is_empty() {
            s.push_str(&format!("note: {}\n", yaml_escape(v)));
        }
    }
    s.push_str("---\n\n");
    s.push_str(&format!("# {}\n\n", args.title));
    s.push_str(args.content.trim());
    s.push('\n');
    s
}

fn yaml_escape(v: &str) -> String {
    // Conservative quoting: if contains : # [ ] { } , & * ? | > <%! ' \" or starts with - or whitespace, quote.
    let needs_quote = v.contains(|c: char| ":#[]{},&*?|<>%!'\"\n".contains(c))
        || v.starts_with('-')
        || v.starts_with(' ')
        || v.is_empty();
    if needs_quote {
        let escaped = v
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', " ");
        format!("\"{}\"", escaped)
    } else {
        v.to_string()
    }
}

pub fn save_article(args: SaveArticleArgs) -> Result<ArticleMeta> {
    let dir = articles_root();
    ensure_dir(&dir)?;
    let safe = safe_filename(&args.title);
    let target = if args.overwrite {
        dir.join(format!("{safe}.md"))
    } else {
        unique_path(&dir, &safe)
    };
    let created_at = now_iso8601();
    let body = render_frontmatter_and_body(&args, args.content.chars().count(), &created_at);
    std::fs::write(&target, body).with_context(|| format!("write {}", target.display()))?;

    Ok(ArticleMeta {
        title: args.title,
        filename: target
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        path: target.to_string_lossy().into_owned(),
        source_audio: args.source_audio,
        source_stem: args.source_stem,
        duration_seconds: args.duration_seconds,
        char_count: args.content.chars().count(),
        model: args.model,
        based_on: args.based_on,
        tags: args.tags.unwrap_or_default(),
        note: args.note,
        created_at: created_at.clone(),
        modified_at: created_at,
    })
}

pub fn list_articles() -> Result<Vec<ArticleMeta>> {
    let root = articles_root();
    if !root.exists() {
        return Ok(vec![]);
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&root)? {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        let raw = match std::fs::read_to_string(&path) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let meta = parse_frontmatter(&raw, &path);
        out.push(meta);
    }
    out.sort_by(|a, b| b.modified_at.cmp(&a.modified_at));
    Ok(out)
}

pub fn delete_article(filename: &str) -> Result<()> {
    let root = articles_root();
    let path = root.join(filename);
    if !path.starts_with(&root) {
        anyhow::bail!("path escapes articles root");
    }
    if path.exists() {
        std::fs::remove_file(&path).with_context(|| format!("rm {}", path.display()))?;
    }
    Ok(())
}

pub fn rename_article(old_filename: &str, new_title: &str) -> Result<ArticleMeta> {
    let root = articles_root();
    let src = root.join(old_filename);
    if !src.exists() {
        anyhow::bail!("article not found: {}", src.display());
    }
    let safe = safe_filename(new_title);
    let dst = unique_path(&root, &safe);
    let raw = std::fs::read_to_string(&src)?;
    // Update title in frontmatter (best-effort regex-free find/replace on first `title:` line)
    let mut updated = String::with_capacity(raw.len());
    let mut in_fm = false;
    let mut fm_seen = 0;
    let mut title_replaced = false;
    for (i, line) in raw.split_inclusive('\n').enumerate() {
        if i == 0 && line.starts_with("---") {
            in_fm = true;
            fm_seen += 1;
            updated.push_str(line);
            continue;
        }
        if in_fm && line.starts_with("---") {
            in_fm = false;
            fm_seen += 1;
            updated.push_str(line);
            continue;
        }
        if in_fm && !title_replaced && line.starts_with("title:") {
            updated.push_str(&format!("title: {}\n", yaml_escape(new_title)));
            title_replaced = true;
            continue;
        }
        if !in_fm && fm_seen >= 2 && line.starts_with("# ") && !title_replaced {
            // Already past frontmatter; replace H1 if no title in fm (shouldn't normally happen)
            updated.push_str(&format!("# {}\n", new_title));
            title_replaced = true;
            continue;
        }
        // Also replace H1 line that follows frontmatter even after title: was replaced
        if !in_fm && fm_seen >= 2 && line.starts_with("# ") {
            updated.push_str(&format!("# {}\n", new_title));
            continue;
        }
        updated.push_str(line);
    }
    std::fs::write(&dst, updated)?;
    std::fs::remove_file(&src)?;
    Ok(parse_frontmatter(&std::fs::read_to_string(&dst)?, &dst))
}

fn parse_frontmatter(raw: &str, path: &Path) -> ArticleMeta {
    let mut title = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut source_audio = None;
    let mut source_stem = None;
    let mut duration_seconds = None;
    let mut char_count = raw.chars().count();
    let mut model = None;
    let mut based_on = None;
    let mut tags: Vec<String> = vec![];
    let mut note = None;
    let mut created_at = String::new();

    if raw.starts_with("---\n") {
        if let Some(end) = raw[4..].find("\n---\n") {
            let fm = &raw[4..4 + end];
            for line in fm.lines() {
                if let Some(rest) = line.strip_prefix("title:") {
                    title = unquote(rest.trim());
                } else if let Some(rest) = line.strip_prefix("source_audio:") {
                    source_audio = Some(unquote(rest.trim()));
                } else if let Some(rest) = line.strip_prefix("source_stem:") {
                    source_stem = Some(unquote(rest.trim()));
                } else if let Some(rest) = line.strip_prefix("duration_seconds:") {
                    duration_seconds = rest.trim().parse::<f64>().ok();
                } else if let Some(rest) = line.strip_prefix("char_count:") {
                    if let Ok(n) = rest.trim().parse::<usize>() {
                        char_count = n;
                    }
                } else if let Some(rest) = line.strip_prefix("model:") {
                    model = Some(unquote(rest.trim()));
                } else if let Some(rest) = line.strip_prefix("based_on:") {
                    based_on = Some(unquote(rest.trim()));
                } else if let Some(rest) = line.strip_prefix("tags:") {
                    let s = rest.trim();
                    if let Some(inner) = s.strip_prefix('[').and_then(|x| x.strip_suffix(']')) {
                        tags = inner
                            .split(',')
                            .map(|t| unquote(t.trim()))
                            .filter(|t| !t.is_empty())
                            .collect();
                    }
                } else if let Some(rest) = line.strip_prefix("created_at:") {
                    created_at = unquote(rest.trim());
                } else if let Some(rest) = line.strip_prefix("note:") {
                    note = Some(unquote(rest.trim()));
                }
            }
        }
    }
    let modified_at = std::fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| {
            let secs = d.as_secs() as i64;
            let days = secs / 86400 + 719162;
            let (y, mo, da) = days_to_ymd(days);
            let h = (secs % 86400) / 3600;
            let m = (secs % 3600) / 60;
            let s = secs % 60;
            format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo, da, h, m, s)
        })
        .unwrap_or_default();

    ArticleMeta {
        title,
        filename: path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        path: path.to_string_lossy().into_owned(),
        source_audio,
        source_stem,
        duration_seconds,
        char_count,
        model,
        based_on,
        tags,
        note,
        created_at: if created_at.is_empty() {
            modified_at.clone()
        } else {
            created_at
        },
        modified_at,
    }
}

fn unquote(s: &str) -> String {
    let s = s.trim();
    if (s.starts_with('"') && s.ends_with('"') && s.len() >= 2)
        || (s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2)
    {
        let inner = &s[1..s.len() - 1];
        return inner.replace("\\\"", "\"").replace("\\\\", "\\");
    }
    s.to_string()
}

pub fn read_article(filename: &str) -> Result<String> {
    let root = articles_root();
    let path = root.join(filename);
    if !path.starts_with(&root) {
        anyhow::bail!("path escapes articles root");
    }
    std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))
}
