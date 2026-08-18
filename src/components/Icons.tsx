// Codicon-style SVG icons. 16x16 viewBox, currentColor stroke/fill.
// 灵感:VSCode codicon set。所有图标默认 stroke-based 外轮廓,适配 dark theme。

import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

const Base = ({
  size = 16,
  children,
  ...rest
}: IconProps & { children: React.ReactNode }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 16 16"
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.25"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden
    {...rest}
  >
    {children}
  </svg>
);

export const ChevronDown = (p: IconProps) => (
  <Base {...p}><path d="M3 5.5l5 5 5-5" /></Base>
);
export const ChevronRight = (p: IconProps) => (
  <Base {...p}><path d="M5.5 3l5 5-5 5" /></Base>
);

export const Settings = (p: IconProps) => (
  <Base {...p}>
    <circle cx="8" cy="8" r="2" />
    <path d="M8 1.5v2M8 12.5v2M14.5 8h-2M3.5 8h-2M12.6 3.4l-1.4 1.4M4.8 11.2l-1.4 1.4M12.6 12.6l-1.4-1.4M4.8 4.8L3.4 3.4" />
  </Base>
);

export const FileAdd = (p: IconProps) => (
  <Base {...p}>
    <path d="M9 1.5H4.5A1.5 1.5 0 003 3v10a1.5 1.5 0 001.5 1.5H11A1.5 1.5 0 0012.5 13V5L9 1.5z" />
    <path d="M9 1.5V5h3.5" />
    <path d="M7.5 8.5v3M6 10h3" />
  </Base>
);

export const ListIcon = (p: IconProps) => (
  <Base {...p}>
    <path d="M5 4h9M5 8h9M5 12h9" />
    <circle cx="2.5" cy="4" r="0.6" fill="currentColor" stroke="none" />
    <circle cx="2.5" cy="8" r="0.6" fill="currentColor" stroke="none" />
    <circle cx="2.5" cy="12" r="0.6" fill="currentColor" stroke="none" />
  </Base>
);

export const Archive = (p: IconProps) => (
  <Base {...p}>
    <rect x="2" y="3" width="12" height="3" rx="0.5" />
    <path d="M3 6v6.5A1.5 1.5 0 004.5 14h7a1.5 1.5 0 001.5-1.5V6" />
    <path d="M6.5 9h3" />
  </Base>
);

export const FileText = (p: IconProps) => (
  <Base {...p}>
    <path d="M9 1.5H4.5A1.5 1.5 0 003 3v10a1.5 1.5 0 001.5 1.5H11A1.5 1.5 0 0012.5 13V5L9 1.5z" />
    <path d="M9 1.5V5h3.5" />
    <path d="M5.5 8h5M5.5 10h5M5.5 12h3" />
  </Base>
);

export const Pencil = (p: IconProps) => (
  <Base {...p}>
    <path d="M11.5 2.5l2 2-8 8H3.5v-2l8-8z" />
    <path d="M10 4l2 2" />
  </Base>
);

export const Article = (p: IconProps) => (
  <Base {...p}>
    <rect x="2" y="2.5" width="12" height="11" rx="1" />
    <path d="M4 5.5h8M4 8h8M4 10.5h6" />
  </Base>
);

export const Pause = (p: IconProps) => (
  <Base {...p}>
    <path d="M5.5 3v10M10.5 3v10" />
  </Base>
);

export const Play = (p: IconProps) => (
  <Base {...p}>
    <path d="M4 3l9 5-9 5V3z" fill="currentColor" />
  </Base>
);

export const Close = (p: IconProps) => (
  <Base {...p}><path d="M3.5 3.5l9 9M12.5 3.5l-9 9" /></Base>
);

export const Trash = (p: IconProps) => (
  <Base {...p}>
    <path d="M2.5 4h11M5.5 4V2.5h5V4M4 4l.7 9.5a1 1 0 001 .9h4.6a1 1 0 001-.9L12 4" />
    <path d="M6.5 7v5M9.5 7v5" />
  </Base>
);

export const Refresh = (p: IconProps) => (
  <Base {...p}>
    <path d="M13.5 8a5.5 5.5 0 11-1.6-3.9" />
    <path d="M13.5 2v2.5H11" />
  </Base>
);

export const FolderOpen = (p: IconProps) => (
  <Base {...p}>
    <path d="M2 5l1-2h4l1 2h6v7.5a1 1 0 01-1 1H3a1 1 0 01-1-1V5z" />
  </Base>
);

export const Check = (p: IconProps) => (
  <Base {...p}><path d="M3 8.5l3.5 3.5L13 5" /></Base>
);

export const Mic = (p: IconProps) => (
  <Base {...p}>
    <rect x="6" y="2" width="4" height="8" rx="2" />
    <path d="M3.5 8a4.5 4.5 0 009 0M8 12.5v2M5.5 14.5h5" />
  </Base>
);

export const Hourglass = (p: IconProps) => (
  <Base {...p}>
    <path d="M4 2h8M4 14h8M4 2c0 3 4 4 4 6S4 11 4 14M12 2c0 3-4 4-4 6s4 3 4 6" />
  </Base>
);

export const Sparkle = (p: IconProps) => (
  <Base {...p}>
    <path d="M8 2v3M8 11v3M2 8h3M11 8h3M3.8 3.8l2 2M10.2 10.2l2 2M3.8 12.2l2-2M10.2 5.8l2-2" />
  </Base>
);

export const Warning = (p: IconProps) => (
  <Base {...p}>
    <path d="M8 2L1.5 13.5h13L8 2z" />
    <path d="M8 6.5v3M8 11.5v.01" strokeWidth="1.5" />
  </Base>
);

export const Info = (p: IconProps) => (
  <Base {...p}>
    <circle cx="8" cy="8" r="6" />
    <path d="M8 7v4M8 5v.01" strokeWidth="1.5" />
  </Base>
);

export const Lock = (p: IconProps) => (
  <Base {...p}>
    <rect x="3.5" y="7" width="9" height="7" rx="1" />
    <path d="M5.5 7V4.5a2.5 2.5 0 015 0V7" />
  </Base>
);

export const Copy = (p: IconProps) => (
  <Base {...p}>
    <rect x="5" y="2.5" width="8.5" height="9" rx="1" />
    <path d="M5 5.5H3.5a1 1 0 00-1 1v6.5a1 1 0 001 1h6.5a1 1 0 001-1V12" />
  </Base>
);

export const Download = (p: IconProps) => (
  <Base {...p}>
    <path d="M8 2v8M5 7l3 3 3-3M2.5 13.5h11" />
  </Base>
);

export const Globe = (p: IconProps) => (
  <Base {...p}>
    <circle cx="8" cy="8" r="6" />
    <path d="M2 8h12M8 2a10 10 0 010 12M8 2a10 10 0 000 12" />
  </Base>
);
