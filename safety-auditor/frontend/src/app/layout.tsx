import type { Metadata } from "next";
import "./globals.css";
import { Nav } from "@/components/nav";

export const metadata: Metadata = {
  title: "Mental Health LLM Safety Auditor",
  description: "Compare Single LLM vs Agent Structure for mental health counseling safety",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body className="min-h-screen bg-surface antialiased">
        <Nav />
        <main>{children}</main>
      </body>
    </html>
  );
}
