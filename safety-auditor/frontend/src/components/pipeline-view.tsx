import type { PipelineStep } from "@/lib/api";

export function PipelineView({ steps, structure }: { steps: PipelineStep[]; structure: string }) {
  return (
    <div className="space-y-3">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Pipeline · {structure}</p>
      <div className="space-y-2">
        {steps.map((s, i) => (
          <div key={s.agent}>
            {i > 0 && <div className="ml-4 h-4 border-l-2 border-accent/30" />}
            <div className="rounded-lg border border-border bg-surface p-3">
              <div className="flex items-center justify-between">
                <span className="font-medium text-slate-800">{s.agent}</span>
                <span className="text-xs text-slate-500">{s.runtime_seconds}s</span>
              </div>
              <p className="mt-1 text-xs text-slate-500">{s.prompt_summary}</p>
              <p className="mt-1 text-sm text-slate-700">{s.output_summary}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
