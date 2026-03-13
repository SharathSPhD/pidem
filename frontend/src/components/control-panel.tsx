"use client";

export interface ControlPanelProps {
  children: React.ReactNode;
  title?: string;
}

export function ControlPanel({ children, title }: ControlPanelProps) {
  return (
    <div>
      {title && (
        <h3 className="mb-3 text-sm font-medium text-slate-500">{title}</h3>
      )}
      <div className="flex flex-wrap gap-6">{children}</div>
    </div>
  );
}
