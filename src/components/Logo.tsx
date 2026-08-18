// 简化版图标(用于头部 inline SVG,无背景方块)。
// 完整版(带圆角方块背景)在 src-tauri/icons/icon.svg。

type Props = { size?: number; className?: string };

export default function Logo({ size = 28, className }: Props) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="LocalScribe"
    >
      <defs>
        <linearGradient id="ls-bg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#6366f1" />
          <stop offset="100%" stopColor="#8b5cf6" />
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="14" fill="url(#ls-bg)" />
      {/* sound waves */}
      <g fill="none" stroke="#fff" strokeLinecap="round">
        <path d="M 14 25 Q 9 32 14 39" strokeWidth="2.5" opacity="0.4" />
        <path d="M 20 28 Q 17 32 20 36" strokeWidth="2.5" opacity="0.7" />
      </g>
      {/* mic body */}
      <rect x="32" y="14" width="14" height="28" rx="7" fill="#fff" />
      {/* yoke */}
      <path
        d="M 28 32 v 6 a 11 11 0 0 0 22 0 v -6"
        fill="none"
        stroke="#e2e8f0"
        strokeWidth="3"
        strokeLinecap="round"
      />
      {/* stem + base */}
      <line x1="39" y1="49" x2="39" y2="56" stroke="#e2e8f0" strokeWidth="3" strokeLinecap="round" />
      <line x1="33" y1="56" x2="45" y2="56" stroke="#e2e8f0" strokeWidth="3" strokeLinecap="round" />
      {/* lock badge */}
      <g transform="translate(50, 50)">
        <circle r="8" fill="#0f172a" />
        <path
          d="M -2 -1 v -1.5 a 2 2 0 0 1 4 0 v 1.5"
          fill="none"
          stroke="#fff"
          strokeWidth="1.2"
          strokeLinecap="round"
        />
        <rect x="-3" y="-1" width="6" height="5" rx="1" fill="#fff" />
      </g>
    </svg>
  );
}
