import Link from "next/link";

export function Nav() {
  return (
    <header className="sticky top-0 z-50 border-b border-border bg-white/90 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4">
        <Link href="/" className="flex items-center gap-2 font-semibold text-slate-900">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-sm text-white">
            MH
          </span>
          Mental Health LLM Safety Auditor
        </Link>
        <nav className="flex gap-4 text-sm">
          <Link href="/" className="text-slate-600 hover:text-accent">
            Home
          </Link>
          <Link href="/compare" className="font-medium text-accent">
            Compare
          </Link>
        </nav>
      </div>
    </header>
  );
}
