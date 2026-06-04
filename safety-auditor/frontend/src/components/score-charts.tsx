"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { JudgeScores } from "@/lib/api";
import { METRICS } from "@/lib/api";

type Props = {
  plain: JudgeScores;
  agent: JudgeScores;
};

export function ScoreTable({ plain, agent }: Props) {
  return (
    <div className="overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-sm">
        <thead className="bg-surface">
          <tr>
            <th className="px-3 py-2 text-left">Metric</th>
            <th className="px-3 py-2 text-right">Single</th>
            <th className="px-3 py-2 text-right">Agent</th>
            <th className="px-3 py-2 text-right">Delta</th>
          </tr>
        </thead>
        <tbody>
          {METRICS.map((m) => {
            const p = plain[m];
            const a = agent[m];
            const d = +(a - p).toFixed(1);
            return (
              <tr key={m} className="border-t border-border">
                <td className="px-3 py-2 capitalize">{m}</td>
                <td className="px-3 py-2 text-right">{p}</td>
                <td className="px-3 py-2 text-right font-medium text-accent">{a}</td>
                <td
                  className={`px-3 py-2 text-right ${d >= 0 ? "text-emerald-600" : "text-rose-600"}`}
                >
                  {d >= 0 ? `+${d}` : d}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function ScoreBarChart({ plain, agent }: Props) {
  const data = METRICS.map((m) => ({
    metric: m,
    Single: plain[m],
    Agent: agent[m],
  }));
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis dataKey="metric" tick={{ fontSize: 11 }} />
          <YAxis domain={[1, 5]} tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend />
          <Bar dataKey="Single" fill="#94a3b8" radius={[4, 4, 0, 0]} />
          <Bar dataKey="Agent" fill="#7c3aed" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ScoreRadarChart({ plain, agent }: Props) {
  const data = METRICS.map((m) => ({
    metric: m,
    Single: plain[m],
    Agent: agent[m],
    fullMark: 5,
  }));
  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <RadarChart data={data}>
          <PolarGrid />
          <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10 }} />
          <Radar name="Single" dataKey="Single" stroke="#94a3b8" fill="#94a3b8" fillOpacity={0.3} />
          <Radar name="Agent" dataKey="Agent" stroke="#7c3aed" fill="#7c3aed" fillOpacity={0.35} />
          <Legend />
          <Tooltip />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
