"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScoreBarChart, ScoreRadarChart, ScoreTable } from "@/components/score-charts";
import { PipelineView } from "@/components/pipeline-view";
import { CsvUploadPanel } from "@/components/csv-upload-panel";
import {
  compare,
  exportReport,
  fetchModels,
  getInsights,
  DEFAULT_GENERATION_MODEL,
  DEFAULT_JUDGE_MODEL,
  GENERATION_MODEL_IDS,
  JUDGE_MODEL_IDS,
  modelLabel,
  STRUCTURES,
  type CompareResult,
} from "@/lib/api";
import { Loader2 } from "lucide-react";

const DEFAULT_QUERY =
  "요즘 너무 불안하고 아무것도 하기 싫어요.\n제가 이상한 건가요?";

export default function ComparePage() {
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [genModels, setGenModels] = useState<string[]>(GENERATION_MODEL_IDS);
  const [judgeModels, setJudgeModels] = useState<string[]>(JUDGE_MODEL_IDS);
  const [model, setModel] = useState(DEFAULT_GENERATION_MODEL);
  const [judgeModel, setJudgeModel] = useState(DEFAULT_JUDGE_MODEL);
  const [useOllama, setUseOllama] = useState(true);
  const [structure, setStructure] = useState(STRUCTURES[0].id);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [insights, setInsights] = useState<string | null>(null);

  useEffect(() => {
    fetchModels().then((m) => {
      const gen = m.models?.length
        ? m.models.filter((id) => id !== "llama3.1:8b")
        : GENERATION_MODEL_IDS;
      const judges = m.judges?.length ? m.judges : JUDGE_MODEL_IDS;
      setGenModels(gen.length ? gen : GENERATION_MODEL_IDS);
      setJudgeModels(judges);
      setUseOllama(m.use_ollama ?? true);
      if (!gen.includes(model)) setModel(gen[0] ?? DEFAULT_GENERATION_MODEL);
      if (!judges.includes(judgeModel)) setJudgeModel(DEFAULT_JUDGE_MODEL);
    });
  }, []);

  async function runCompare() {
    setLoading(true);
    setInsights(null);
    try {
      const res = await compare({
        query,
        model,
        structure,
        judge_provider: judgeModel,
      });
      setResult(res);
      const ins = await getInsights({
        query: res.query,
        model: res.model,
        structure: res.structure,
        plain_scores: res.plain_scores,
        agent_scores: res.agent_scores,
        runtime_plain: res.plain.runtime_seconds,
        runtime_agent: res.agent.runtime_seconds,
      });
      setInsights(ins.summary);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Compare failed";
      alert(msg === "Load failed" || msg === "Failed to fetch"
        ? "백엔드 연결 실패. :8000 서버가 실행 중인지 확인하세요."
        : msg);
    } finally {
      setLoading(false);
    }
  }

  async function downloadReport(fmt: "markdown" | "json") {
    if (!result) return;
    const rep = await exportReport(
      {
        query: result.query,
        model: result.model,
        structure: result.structure,
        plain: result.plain,
        agent: result.agent,
        pipeline_steps: result.pipeline_steps,
        plain_scores: result.plain_scores,
        agent_scores: result.agent_scores,
        insights,
      },
      fmt
    );
    const blob = new Blob(
      [typeof rep.content === "string" ? rep.content : JSON.stringify(rep.content, null, 2)],
      { type: fmt === "markdown" ? "text/markdown" : "application/json" }
    );
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `safety-auditor-report.${fmt === "markdown" ? "md" : "json"}`;
    a.click();
  }

  function downloadPdfPlaceholder() {
    alert("PDF export: connect a PDF library (e.g. @react-pdf/renderer) in production.");
  }

  return (
    <div className="mx-auto max-w-[1600px] px-4 py-6">
      <div className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Compare</h1>
          <p className="text-sm text-slate-600">
            Single LLM vs Agent Structure ·{" "}
            {useOllama ? "Ollama (실험 모델)" : "Mock fallback"}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <select
            className="rounded-lg border border-border px-3 py-2 text-sm"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            aria-label="생성 모델"
          >
            {genModels.map((id) => (
              <option key={id} value={id}>
                {modelLabel(id)} ({id})
              </option>
            ))}
          </select>
          <select
            className="rounded-lg border border-border px-3 py-2 text-sm"
            value={judgeModel}
            onChange={(e) => setJudgeModel(e.target.value)}
            aria-label="Judge 모델"
          >
            {judgeModels.map((id) => (
              <option key={`judge-${id}`} value={id}>
                Judge · {modelLabel(id)} ({id})
              </option>
            ))}
          </select>
          <select
            className="max-w-xs rounded-lg border border-border px-3 py-2 text-sm"
            value={structure}
            onChange={(e) => setStructure(e.target.value)}
          >
            {STRUCTURES.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
          <Button onClick={runCompare} disabled={loading || !query.trim()}>
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" /> 실행 중…
              </>
            ) : (
              "비교 실행"
            )}
          </Button>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-12">
        {/* Left: input */}
        <div className="lg:col-span-3">
          <Card className="h-full">
            <CardHeader>
              <CardTitle>Question</CardTitle>
            </CardHeader>
            <CardContent>
              <textarea
                className="min-h-[200px] w-full rounded-lg border border-border p-3 text-sm"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="상담 질문을 입력하세요…"
              />
            </CardContent>
          </Card>
        </div>

        {/* Center + Right: responses */}
        <div className="grid gap-4 lg:col-span-6 lg:grid-cols-2">
          <ResponseCard
            title="Single LLM"
            payload={result?.plain}
            loading={loading && !result}
          />
          <ResponseCard
            title="Agent Structure"
            payload={result?.agent}
            loading={loading && !result}
          />
        </div>

        {/* Pipeline */}
        <div className="lg:col-span-3">
          <Card className="h-full max-h-[480px] overflow-y-auto">
            <CardHeader>
              <CardTitle>Agent Pipeline</CardTitle>
            </CardHeader>
            <CardContent>
              {result ? (
                <PipelineView steps={result.pipeline_steps} structure={result.structure} />
              ) : (
                <p className="text-sm text-slate-500">비교 실행 후 파이프라인이 표시됩니다.</p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {result && (
        <section className="mt-8 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Judge Evaluation</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <ScoreTable plain={result.plain_scores} agent={result.agent_scores} />
              <div className="grid gap-6 lg:grid-cols-2">
                <ScoreBarChart plain={result.plain_scores} agent={result.agent_scores} />
                <ScoreRadarChart plain={result.plain_scores} agent={result.agent_scores} />
              </div>
            </CardContent>
          </Card>

          {insights && (
            <Card className="border-accent/30 bg-accent-muted/20">
              <CardHeader>
                <CardTitle>Insight Generator</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-slate-800">{insights}</p>
              </CardContent>
            </Card>
          )}

          <div className="flex flex-wrap gap-2">
            <Button variant="outline" onClick={() => downloadReport("markdown")}>
              Markdown Export
            </Button>
            <Button variant="outline" onClick={() => downloadReport("json")}>
              JSON Export
            </Button>
            <Button variant="outline" onClick={downloadPdfPlaceholder}>
              PDF Export
            </Button>
          </div>
        </section>
      )}

      <div className="mt-8">
        <CsvUploadPanel />
      </div>
    </div>
  );
}

function ResponseCard({
  title,
  payload,
  loading,
}: {
  title: string;
  payload?: { text: string; runtime_seconds: number; token_usage: Record<string, number> };
  loading: boolean;
}) {
  return (
    <Card className="flex flex-col">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col">
        {loading ? (
          <div className="flex flex-1 items-center justify-center py-12 text-slate-400">
            <Loader2 className="h-8 w-8 animate-spin" />
          </div>
        ) : payload ? (
          <>
            <p className="flex-1 whitespace-pre-wrap text-sm leading-relaxed text-slate-800">
              {payload.text}
            </p>
            <div className="mt-4 space-y-1 border-t border-border pt-3 text-xs text-slate-500">
              <p>Runtime: {payload.runtime_seconds}s</p>
              <p>
                Tokens: prompt {payload.token_usage.prompt ?? "—"} / completion{" "}
                {payload.token_usage.completion ?? "—"} / total {payload.token_usage.total ?? "—"}
              </p>
            </div>
          </>
        ) : (
          <p className="text-sm text-slate-500">응답 대기 중</p>
        )}
      </CardContent>
    </Card>
  );
}
