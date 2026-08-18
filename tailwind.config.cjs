/** @type {import('tailwindcss').Config} */
// 配色对齐 VSCode Dark+ (来自 vscode/src/vs/workbench/common/theme.ts)
module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // 主背景 — VSCode editor.background
        editor: "#1e1e1e",
        // 侧栏背景 — sideBar.background / panel.background
        sidebar: "#252526",
        // 标题栏背景 — titleBar.activeBackground / activityBar.background
        titlebar: "#333333",
        // tab 未激活背景 — editorGroupHeader.tabsBackground
        tabbar: "#252526",
        // 输入框背景 — input.background
        input: "#3c3c3c",
        // 状态栏 — statusBar.background (active focused)
        statusbar: "#007acc",
        // 边框/分隔 — panel.border / 3c3c3c
        border: {
          DEFAULT: "#3c3c3c",
          hover: "#505050",
        },
        // hover bg in lists — list.hoverBackground
        hover: "#2a2d2e",
        // selection bg — list.activeSelectionBackground
        selected: "#04395e",
        // 文字
        fg: "#cccccc",
        "fg-dim": "#9d9d9d",
        "fg-mute": "#6c6c6c",
        // accent — focusBorder / activityBarBadge
        accent: {
          DEFAULT: "#007fd4",
          hover: "#0098ff",
        },
        // status colors
        warn: "#cca700",
        err: "#f48771",
        ok: "#89d185",
      },
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "PingFang SC",
          "Microsoft YaHei",
          "sans-serif",
        ],
        mono: [
          "Menlo",
          "Monaco",
          "Consolas",
          "Droid Sans Mono",
          "ui-monospace",
          "monospace",
        ],
      },
      fontSize: {
        // VSCode 默认 13px / 12px / 11px 三档
        "ui-sm": ["11px", "16px"],
        "ui": ["13px", "20px"],
        "ui-lg": ["14px", "20px"],
      },
    },
  },
  plugins: [],
};
