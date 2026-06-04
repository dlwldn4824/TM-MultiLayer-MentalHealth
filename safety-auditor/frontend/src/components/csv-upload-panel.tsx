"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { getCsvComparison, uploadCsv } from "@/lib/api";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const LABELS = ["D_OFF", "A_OFF", "Router", "Bidirectional"];

export function CsvUploadPanel() {
  const [chartData, setChartData] = useState<{ metric: string; [k: string]: string | number }[]>([]);
  const [loading, setLoading] = useState(false);

  async function onUpload(label: string, file: File) {
    setLoading(true);
    try {
      await uploadCsv(label, file);
      const cmp = await getCsvComparison();
      const series = cmp.comparison_chart as { metric: string; series: { label: string; value: number }[] }[];
      const rows = series.map((s) => {
        const row: { metric: string; [k: string]: string | number } = { metric: s.metric };
        for (const pt of s.series) row[pt.label] = pt.value;
        return row;
      });
      setChartData(rows);
    } catch (e) {
      alert(e instanceof Error ? e.message : "CSV 업로드 실패");
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Experiment CSV Upload</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-2">
          {LABELS.map((label) => (
            <label key={label} className="cursor-pointer">
              <input
                type="file"
                accept=".csv"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) onUpload(label, f);
                }}
              />
              <span className="inline-flex rounded-lg border border-dashed border-accent/40 px-3 py-2 text-xs hover:bg-accent-muted">
                {label}
              </span>
            </label>
          ))}
        </div>
        {loading && <p className="text-sm text-slate-500">Uploading…</p>}
        {chartData.length > 0 && (
          <div className="h-56">
            <ResponsiveContainer>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="metric" tick={{ fontSize: 10 }} />
                <YAxis domain={[1, 5]} />
                <Tooltip />
                <Legend />
                {LABELS.map((l, i) => (
                  <Bar
                    key={l}
                    dataKey={l}
                    fill={["#7c3aed", "#a78bfa", "#6366f1", "#8b5cf6"][i]}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        <Button variant="outline" onClick={async () => {
          const cmp = await getCsvComparison();
          console.log(cmp);
        }}>
          Refresh comparison
        </Button>
      </CardContent>
    </Card>
  );
}
