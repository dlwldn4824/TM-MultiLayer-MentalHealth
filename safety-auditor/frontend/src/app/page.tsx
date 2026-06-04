import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Shield, GitCompare, BarChart3, FileText } from "lucide-react";

export default function HomePage() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-16">
      <div className="text-center">
        <p className="text-sm font-medium text-accent">Clinical AI Safety Research</p>
        <h1 className="mt-2 text-4xl font-bold tracking-tight text-slate-900">
          Mental Health LLM Safety Auditor
        </h1>
        <p className="mx-auto mt-4 max-w-2xl text-lg text-slate-600">
          Single LLM(단일 호출) 응답과 Agent Structure 응답을 동일 질문으로 비교하고,
          Judge가 Safety·Empathy·Trust 등을 평가합니다.
        </p>
        <Link href="/compare" className="mt-8 inline-block">
          <Button className="px-8 py-3 text-base">비교 시작하기</Button>
        </Link>
      </div>

      <div className="mt-16 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
        {[
          { icon: GitCompare, title: "Side-by-side Compare", desc: "Single vs Agent 실시간 비교" },
          { icon: Shield, title: "Safety Judge", desc: "6차원 1–5 점수 자동 평가" },
          { icon: BarChart3, title: "Charts", desc: "Bar·Radar·Delta 테이블" },
          { icon: FileText, title: "Reports", desc: "Markdown / JSON보내기" },
        ].map(({ icon: Icon, title, desc }) => (
          <Card key={title}>
            <CardHeader>
              <Icon className="h-6 w-6 text-accent" />
              <CardTitle className="mt-2 text-base">{title}</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-slate-600">{desc}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card className="mt-12 border-accent/20 bg-accent-muted/30">
        <CardContent className="py-6 text-center text-sm text-slate-700">
          Ollama 로컬 모델 (qwen2.5:7b, llama3.1:8b, gemma2:9b, mistral:7b) · OpenAI 미사용
        </CardContent>
      </Card>
    </div>
  );
}
