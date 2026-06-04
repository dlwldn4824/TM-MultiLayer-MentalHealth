const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type JudgeScores = {
  safety: number;
  empathy: number;
  trust: number;
  helpfulness: number;
  faithfulness: number;
  coherence: number;
};

export type ResponsePayload = {
  text: string;
  runtime_seconds: number;
  token_usage: Record<string, number>;
};

export type PipelineStep = {
  agent: string;
  runtime_seconds: number;
  prompt_summary: string;
  output_summary: string;
};

export type CompareResult = {
  query: string;
  model: string;
  structure: string;
  plain: ResponsePayload;
  agent: ResponsePayload;
  pipeline_steps: PipelineStep[];
  plain_scores: JudgeScores;
  agent_scores: JudgeScores;
  evaluation_id: number | null;
};

async function fetchApi<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || res.statusText);
  }
  return res.json() as Promise<T>;
}

export async function compare(payload: {
  query: string;
  model: string;
  structure: string;
  judge_provider?: string;
}): Promise<CompareResult> {
  return fetchApi("/api/compare", {
    method: "POST",
    body: JSON.stringify({
      judge_provider: "llama3.1:8b",
      ...payload,
    }),
  });
}

export async function getInsights(payload: {
  query: string;
  model: string;
  structure: string;
  plain_scores: JudgeScores;
  agent_scores: JudgeScores;
  runtime_plain: number;
  runtime_agent: number;
}): Promise<{ summary: string; bullets: string[] }> {
  return fetchApi("/api/insights", { method: "POST", body: JSON.stringify(payload) });
}

export async function exportReport(
  payload: Record<string, unknown>,
  format: "markdown" | "json"
): Promise<{ format: string; content: string | Record<string, unknown> }> {
  return fetchApi("/api/report", {
    method: "POST",
    body: JSON.stringify({ ...payload, format }),
  });
}

export async function uploadCsv(label: string, file: File) {
  const form = new FormData();
  form.append("label", label);
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/upload-csv`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getCsvComparison() {
  return fetchApi<{ uploads: unknown[]; comparison_chart: unknown[] }>("/api/csv-comparison");
}

export const STRUCTURES = [
  { id: "emotion_strategy_response_safety", label: "Emotion → Strategy → Response → Safety" },
  { id: "emotion_response_safety", label: "Emotion → Response → Safety" },
  { id: "router", label: "Router" },
  { id: "bidirectional", label: "Bidirectional" },
];

/** Same four Ollama models as TM-MultiLayer-MentalHealth experiments */
export const EXPERIMENT_MODEL_OPTIONS = [
  { id: "qwen2.5:7b", label: "Qwen 2.5 7B", role: "generation" as const },
  { id: "llama3.1:8b", label: "Llama 3.1 8B", role: "judge" as const },
  { id: "gemma2:9b", label: "Gemma 2 9B", role: "generation" as const },
  { id: "mistral:7b", label: "Mistral 7B", role: "generation" as const },
];

export const DEFAULT_GENERATION_MODEL = "qwen2.5:7b";
export const DEFAULT_JUDGE_MODEL = "llama3.1:8b";

export const GENERATION_MODEL_IDS = EXPERIMENT_MODEL_OPTIONS.filter(
  (m) => m.role === "generation"
).map((m) => m.id);

export const JUDGE_MODEL_IDS = [
  DEFAULT_JUDGE_MODEL,
  ...EXPERIMENT_MODEL_OPTIONS.map((m) => m.id).filter((id) => id !== DEFAULT_JUDGE_MODEL),
];

export function modelLabel(id: string): string {
  return EXPERIMENT_MODEL_OPTIONS.find((m) => m.id === id)?.label ?? id;
}

export type ModelsResponse = {
  models: string[];
  judges: string[];
  use_ollama: boolean;
  llm_mode: string;
  ollama_available?: Record<string, boolean>;
  model_details?: { id: string; label: string; role: string }[];
};

export async function fetchModels(): Promise<ModelsResponse> {
  try {
    return await fetchApi("/api/models");
  } catch {
    return {
      models: GENERATION_MODEL_IDS,
      judges: JUDGE_MODEL_IDS,
      use_ollama: false,
      llm_mode: "fallback",
    };
  }
}

export const METRICS: (keyof JudgeScores)[] = [
  "safety",
  "empathy",
  "trust",
  "helpfulness",
  "faithfulness",
  "coherence",
];
