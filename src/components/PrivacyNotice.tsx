import { Warning } from "./Icons";

type Props = {
  onAccept: () => void;
  onCancel: () => void;
};

export default function PrivacyNotice({ onAccept, onCancel }: Props) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-sidebar border border-border rounded-sm shadow-2xl max-w-xl w-full m-6">
        <header className="flex items-center gap-2 px-4 py-3 border-b border-border">
          <Warning size={16} className="text-warn" />
          <h2 className="text-ui-lg font-medium text-fg">启用 LLM 校对前的提示</h2>
        </header>
        <div className="px-4 py-4 text-ui leading-relaxed text-fg-dim space-y-2">
          <p>启用后,LocalScribe 会把<strong className="text-fg">转录后的文字</strong>发送到你配置的 API 提供商。</p>
          <ul className="list-disc pl-5 space-y-1">
            <li>原始<strong className="text-fg">音频文件不会上传</strong></li>
            <li><strong className="text-fg">转录文字</strong>会经过第三方服务器</li>
            <li>提供商可能记录或使用这些数据(取决于你与其的合约)</li>
          </ul>
          <p className="pt-2 text-ui-sm text-fg-mute">
            如内容涉及商业机密、医疗、法律或个人隐私,建议保持关闭使用纯离线模式。
          </p>
        </div>
        <div className="flex justify-end gap-2 px-4 py-3 border-t border-border bg-editor/40">
          <button onClick={onCancel} className="btn-ghost">取消</button>
          <button onClick={onAccept} className="btn">我已知晓,启用</button>
        </div>
      </div>
    </div>
  );
}
